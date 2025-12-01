use anyhow::Result;
use edit_prediction::{Direction, EditPrediction, EditPredictionProvider};
use futures::AsyncReadExt;
use gpui::{App, Context, Entity, EntityId, Global, Task};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use language::{Anchor, Buffer, BufferSnapshot, Point};
use serde::{Deserialize, Serialize};
use std::{ops::Range, sync::Arc, time::Duration};
use text::{ToOffset, ToPoint};
use unicode_segmentation::UnicodeSegmentation;

pub const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(75);
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen2.5-coder:7b";

#[derive(Clone, Debug, PartialEq)]
pub enum OllamaConnectionStatus {
    Unknown,
    Connected,
    Error(String),
}

impl Default for OllamaConnectionStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Default)]
struct GlobalOllamaStatus(OllamaConnectionStatus);

impl Global for GlobalOllamaStatus {}

pub fn ollama_connection_status(cx: &App) -> OllamaConnectionStatus {
    cx.try_global::<GlobalOllamaStatus>()
        .map(|status| status.0.clone())
        .unwrap_or_default()
}

fn set_ollama_connection_status(status: OllamaConnectionStatus, cx: &mut App) {
    cx.set_global(GlobalOllamaStatus(status));
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<GenerateOptions>,
}

#[derive(Serialize)]
struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
}

pub struct OllamaCompletionProvider {
    http_client: Arc<dyn HttpClient>,
    api_url: String,
    model: String,
    buffer_id: Option<EntityId>,
    completion_text: Option<String>,
    pending_refresh: Option<Task<Result<()>>>,
    completion_position: Option<Anchor>,
}

impl OllamaCompletionProvider {
    pub fn new(http_client: Arc<dyn HttpClient>) -> Self {
        Self {
            http_client,
            api_url: DEFAULT_OLLAMA_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            buffer_id: None,
            completion_text: None,
            pending_refresh: None,
            completion_position: None,
        }
    }

    pub fn with_url(mut self, url: String) -> Self {
        self.api_url = url;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    fn build_fim_prompt(
        &self,
        snapshot: &BufferSnapshot,
        cursor_position: Anchor,
    ) -> String {
        let cursor_offset = cursor_position.to_offset(snapshot);

        let prefix = snapshot.text_for_range(0..cursor_offset).collect::<String>();
        let suffix = snapshot.text_for_range(cursor_offset..snapshot.len()).collect::<String>();

        format!(
            "<|fim_prefix|>{}<|fim_suffix|>{}<|fim_middle|>",
            prefix, suffix
        )
    }

    async fn fetch_completion(
        http_client: Arc<dyn HttpClient>,
        api_url: String,
        model: String,
        prompt: String,
    ) -> Result<String> {
        let uri = format!("{}/api/generate", api_url);

        let request_body = GenerateRequest {
            model,
            prompt,
            stream: false,
            options: Some(GenerateOptions {
                num_predict: Some(256),
                temperature: Some(0.2),
                stop: Some(vec![
                    "\n\n".to_string(),
                    "<|fim_prefix|>".to_string(),
                    "<|fim_suffix|>".to_string(),
                    "<|fim_middle|>".to_string(),
                ]),
            }),
        };

        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri(uri)
            .header("Content-Type", "application/json")
            .body(AsyncBody::from(serde_json::to_string(&request_body)?))?;

        let mut response = http_client.send(request).await?;

        if response.status().is_success() {
            let mut body = String::new();
            response.body_mut().read_to_string(&mut body).await?;
            let response: GenerateResponse = serde_json::from_str(&body)?;
            Ok(response.response)
        } else {
            let mut body = String::new();
            response.body_mut().read_to_string(&mut body).await?;
            anyhow::bail!("Ollama API error: {} {}", response.status(), body)
        }
    }
}

fn completion_from_text(
    snapshot: BufferSnapshot,
    completion_text: &str,
    position: Anchor,
) -> EditPrediction {
    let cursor_point = position.to_point(&snapshot);
    let end_of_line = snapshot.anchor_after(Point::new(
        cursor_point.row,
        snapshot.line_len(cursor_point.row),
    ));
    let delete_range = position..end_of_line;

    let buffer_text = snapshot.text_for_range(delete_range.clone()).collect::<String>();
    let mut edits: Vec<(Range<Anchor>, Arc<str>)> = Vec::new();

    let completion_graphemes: Vec<&str> = completion_text.graphemes(true).collect();
    let buffer_graphemes: Vec<&str> = buffer_text.graphemes(true).collect();

    let mut offset = position.to_offset(&snapshot);

    let mut i = 0;
    let mut j = 0;
    while i < completion_graphemes.len() && j < buffer_graphemes.len() {
        let k = completion_graphemes[i..]
            .iter()
            .position(|c| *c == buffer_graphemes[j]);
        match k {
            Some(k) => {
                if k != 0 {
                    let offset_anchor = snapshot.anchor_after(offset);
                    let edit = (
                        offset_anchor..offset_anchor,
                        completion_graphemes[i..i + k].join("").into(),
                    );
                    edits.push(edit);
                }
                i += k + 1;
                j += 1;
                offset += buffer_graphemes[j - 1].len();
            }
            None => {
                break;
            }
        }
    }

    if j == buffer_graphemes.len() && i < completion_graphemes.len() {
        let offset_anchor = snapshot.anchor_after(offset);
        let edit_range = offset_anchor..offset_anchor;
        let edit_text = completion_graphemes[i..].join("");
        edits.push((edit_range, edit_text.into()));
    }

    EditPrediction::Local {
        id: None,
        edits,
        edit_preview: None,
    }
}

fn trim_to_end_of_line_unless_leading_newline(text: &str) -> &str {
    if has_leading_newline(text) {
        text
    } else if let Some(i) = text.find('\n') {
        &text[..i]
    } else {
        text
    }
}

fn has_leading_newline(text: &str) -> bool {
    for c in text.chars() {
        if c == '\n' {
            return true;
        }
        if !c.is_whitespace() {
            return false;
        }
    }
    false
}

fn reset_completion_cache(provider: &mut OllamaCompletionProvider) {
    provider.pending_refresh = None;
    provider.completion_text = None;
    provider.completion_position = None;
    provider.buffer_id = None;
}

impl EditPredictionProvider for OllamaCompletionProvider {
    fn name() -> &'static str {
        "ollama"
    }

    fn display_name() -> &'static str {
        "Ollama"
    }

    fn show_completions_in_menu() -> bool {
        true
    }

    fn show_tab_accept_marker() -> bool {
        true
    }

    fn supports_jump_to_edit() -> bool {
        false
    }

    fn is_enabled(&self, _buffer: &Entity<Buffer>, _cursor_position: Anchor, _cx: &App) -> bool {
        true
    }

    fn is_refreshing(&self, _cx: &App) -> bool {
        self.pending_refresh.is_some() && self.completion_text.is_none()
    }

    fn refresh(
        &mut self,
        buffer_handle: Entity<Buffer>,
        cursor_position: Anchor,
        debounce: bool,
        cx: &mut Context<Self>,
    ) {
        if !debounce {
            return;
        }

        reset_completion_cache(self);

        let snapshot = buffer_handle.read(cx).snapshot();
        let prompt = self.build_fim_prompt(&snapshot, cursor_position);

        let http_client = self.http_client.clone();
        let api_url = self.api_url.clone();
        let model = self.model.clone();
        let buffer_id = buffer_handle.entity_id();

        self.pending_refresh = Some(cx.spawn(async move |this, cx| {
            if debounce {
                cx.background_executor().timer(DEBOUNCE_TIMEOUT).await;
            }

            let completion = Self::fetch_completion(
                http_client,
                api_url,
                model,
                prompt,
            ).await;

            match completion {
                Ok(text) => {
                    this.update(cx, |this, cx| {
                        this.completion_text = Some(text);
                        this.completion_position = Some(cursor_position);
                        this.buffer_id = Some(buffer_id);
                        cx.notify();
                    })?;
                    cx.update(|cx| {
                        set_ollama_connection_status(OllamaConnectionStatus::Connected, cx);
                    })?;
                }
                Err(err) => {
                    let error_message = err.to_string();
                    log::warn!("Ollama completion error: {}", error_message);
                    cx.update(|cx| {
                        set_ollama_connection_status(
                            OllamaConnectionStatus::Error(error_message),
                            cx,
                        );
                    })?;
                }
            }
            Ok(())
        }));
    }

    fn cycle(
        &mut self,
        _buffer: Entity<Buffer>,
        _cursor_position: Anchor,
        _direction: Direction,
        _cx: &mut Context<Self>,
    ) {
    }

    fn accept(&mut self, _cx: &mut Context<Self>) {
        reset_completion_cache(self);
    }

    fn discard(&mut self, _cx: &mut Context<Self>) {
        reset_completion_cache(self);
    }

    fn suggest(
        &mut self,
        buffer: &Entity<Buffer>,
        cursor_position: Anchor,
        cx: &mut Context<Self>,
    ) -> Option<EditPrediction> {
        if self.buffer_id != Some(buffer.entity_id()) {
            return None;
        }

        let completion_text = self.completion_text.as_ref()?;

        if let Some(completion_position) = self.completion_position {
            if cursor_position != completion_position {
                return None;
            }
        } else {
            return None;
        }

        let completion_text = trim_to_end_of_line_unless_leading_newline(completion_text);
        let completion_text = completion_text.trim_end();

        if completion_text.trim().is_empty() {
            return None;
        }

        let snapshot = buffer.read(cx).snapshot();
        Some(completion_from_text(snapshot, completion_text, cursor_position))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_leading_newline() {
        assert!(has_leading_newline("\nfoo"));
        assert!(has_leading_newline("  \nfoo"));
        assert!(has_leading_newline("\t\nfoo"));
        assert!(!has_leading_newline("foo\nbar"));
        assert!(!has_leading_newline("foo"));
        assert!(!has_leading_newline(""));
    }

    #[test]
    fn test_trim_to_end_of_line_unless_leading_newline() {
        assert_eq!(
            trim_to_end_of_line_unless_leading_newline("foo\nbar"),
            "foo"
        );
        assert_eq!(
            trim_to_end_of_line_unless_leading_newline("foo bar"),
            "foo bar"
        );
        assert_eq!(
            trim_to_end_of_line_unless_leading_newline("\nfoo\nbar"),
            "\nfoo\nbar"
        );
        assert_eq!(
            trim_to_end_of_line_unless_leading_newline("  \nfoo"),
            "  \nfoo"
        );
    }

    #[test]
    fn test_ollama_connection_status_default() {
        let status = OllamaConnectionStatus::default();
        assert_eq!(status, OllamaConnectionStatus::Unknown);
    }

    #[test]
    fn test_ollama_connection_status_equality() {
        assert_eq!(
            OllamaConnectionStatus::Connected,
            OllamaConnectionStatus::Connected
        );
        assert_eq!(
            OllamaConnectionStatus::Error("test".to_string()),
            OllamaConnectionStatus::Error("test".to_string())
        );
        assert_ne!(
            OllamaConnectionStatus::Connected,
            OllamaConnectionStatus::Unknown
        );
        assert_ne!(
            OllamaConnectionStatus::Error("a".to_string()),
            OllamaConnectionStatus::Error("b".to_string())
        );
    }

    #[test]
    fn test_provider_builder_methods() {
        use http_client::FakeHttpClient;

        let http_client = FakeHttpClient::with_404_response();
        let provider = OllamaCompletionProvider::new(http_client.clone());

        assert_eq!(provider.api_url, DEFAULT_OLLAMA_URL);
        assert_eq!(provider.model, DEFAULT_MODEL);

        let provider = provider
            .with_url("http://custom:8080".to_string())
            .with_model("custom-model".to_string());

        assert_eq!(provider.api_url, "http://custom:8080");
        assert_eq!(provider.model, "custom-model");
    }

    #[test]
    fn test_generate_request_serialization() {
        let request = GenerateRequest {
            model: "test-model".to_string(),
            prompt: "test prompt".to_string(),
            stream: false,
            options: Some(GenerateOptions {
                num_predict: Some(256),
                temperature: Some(0.2),
                stop: Some(vec!["\n\n".to_string()]),
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"test-model\""));
        assert!(json.contains("\"prompt\":\"test prompt\""));
        assert!(json.contains("\"stream\":false"));
        assert!(json.contains("\"num_predict\":256"));
        assert!(json.contains("\"temperature\":0.2"));
    }

    #[test]
    fn test_generate_response_deserialization() {
        let json = r#"{"response": "completed code", "done": true}"#;
        let response: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.response, "completed code");
    }
}
