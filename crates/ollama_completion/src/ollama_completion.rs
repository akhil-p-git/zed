use anyhow::Result;
use collections::HashMap;
use edit_prediction::{Direction, EditPrediction, EditPredictionProvider};
use futures::AsyncReadExt;
use gpui::{App, Context, Entity, EntityId, Global, Task};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use language::{Anchor, Buffer, BufferSnapshot, Point};
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
    time::Duration,
};
use text::{ToOffset, ToPoint};
use unicode_segmentation::UnicodeSegmentation;

pub const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(75);
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen2.5-coder:7b";

/// Maximum bytes for prefix context (code before cursor)
const MAX_PREFIX_BYTES: usize = 4096;
/// Maximum bytes for suffix context (code after cursor)
const MAX_SUFFIX_BYTES: usize = 1024;
/// Maximum number of cached completions
const MAX_CACHE_SIZE: usize = 50;

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

/// Response from Ollama's /api/tags endpoint
#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

/// Cached completion entry
#[derive(Clone)]
struct CachedCompletion {
    prompt_hash: u64,
    completion: String,
}

pub struct OllamaCompletionProvider {
    http_client: Arc<dyn HttpClient>,
    api_url: String,
    model: String,
    buffer_id: Option<EntityId>,
    completion_text: Option<String>,
    pending_refresh: Option<Task<Result<()>>>,
    completion_position: Option<Anchor>,
    /// LRU-style cache: maps prompt hash to completion text
    completion_cache: HashMap<u64, String>,
    /// Order of cache entries for LRU eviction
    cache_order: Vec<u64>,
    /// Whether initial health check has been performed
    health_checked: bool,
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
            completion_cache: HashMap::default(),
            cache_order: Vec::new(),
            health_checked: false,
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

    /// Build FIM prompt with optimized context window
    fn build_fim_prompt(
        &self,
        snapshot: &BufferSnapshot,
        cursor_position: Anchor,
    ) -> String {
        let cursor_offset = cursor_position.to_offset(snapshot);

        // Get full prefix and suffix
        let full_prefix: String = snapshot.text_for_range(0..cursor_offset).collect();
        let full_suffix: String = snapshot.text_for_range(cursor_offset..snapshot.len()).collect();

        // Optimize prefix: take last MAX_PREFIX_BYTES, but try to start at a line boundary
        let prefix = if full_prefix.len() > MAX_PREFIX_BYTES {
            let start = full_prefix.len() - MAX_PREFIX_BYTES;
            // Find next newline after start to get a clean line boundary
            if let Some(newline_offset) = full_prefix[start..].find('\n') {
                &full_prefix[start + newline_offset + 1..]
            } else {
                &full_prefix[start..]
            }
        } else {
            &full_prefix
        };

        // Optimize suffix: take first MAX_SUFFIX_BYTES, but try to end at a line boundary
        let suffix = if full_suffix.len() > MAX_SUFFIX_BYTES {
            // Find last newline before limit to get a clean line boundary
            if let Some(newline_offset) = full_suffix[..MAX_SUFFIX_BYTES].rfind('\n') {
                &full_suffix[..newline_offset + 1]
            } else {
                &full_suffix[..MAX_SUFFIX_BYTES]
            }
        } else {
            &full_suffix
        };

        format!(
            "<|fim_prefix|>{}<|fim_suffix|>{}<|fim_middle|>",
            prefix, suffix
        )
    }

    /// Compute hash of a prompt for caching
    fn hash_prompt(prompt: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached completion if available
    fn get_cached(&self, prompt_hash: u64) -> Option<&String> {
        self.completion_cache.get(&prompt_hash)
    }

    /// Store completion in cache with LRU eviction
    fn cache_completion(&mut self, prompt_hash: u64, completion: String) {
        // Remove if already exists (will re-add at end for LRU)
        if self.completion_cache.contains_key(&prompt_hash) {
            self.cache_order.retain(|&h| h != prompt_hash);
        }

        // Evict oldest if at capacity
        while self.completion_cache.len() >= MAX_CACHE_SIZE {
            if let Some(oldest_hash) = self.cache_order.first().copied() {
                self.completion_cache.remove(&oldest_hash);
                self.cache_order.remove(0);
            } else {
                break;
            }
        }

        self.completion_cache.insert(prompt_hash, completion);
        self.cache_order.push(prompt_hash);
    }

    /// Check if Ollama server is running and model is available
    pub async fn check_health(
        http_client: Arc<dyn HttpClient>,
        api_url: &str,
        model: &str,
    ) -> Result<(), String> {
        let uri = format!("{}/api/tags", api_url);

        let request = match HttpRequest::builder()
            .method(Method::GET)
            .uri(&uri)
            .body(AsyncBody::default())
        {
            Ok(req) => req,
            Err(e) => return Err(format!("Failed to build request: {}", e)),
        };

        let mut response = match http_client.send(request).await {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Connection refused") {
                    return Err("Ollama server is not running. Start with 'ollama serve'.".to_string());
                }
                return Err(format!("Cannot connect to Ollama: {}", error_msg));
            }
        };

        if !response.status().is_success() {
            return Err(format!("Ollama API error: {}", response.status()));
        }

        let mut body = String::new();
        if let Err(e) = response.body_mut().read_to_string(&mut body).await {
            return Err(format!("Failed to read response: {}", e));
        }

        let tags: TagsResponse = match serde_json::from_str(&body) {
            Ok(t) => t,
            Err(e) => return Err(format!("Invalid response from Ollama: {}", e)),
        };

        // Check if the configured model is available
        // Model names can be "model:tag" or just "model"
        let model_base = model.split(':').next().unwrap_or(model);
        let model_available = tags.models.iter().any(|m| {
            let available_base = m.name.split(':').next().unwrap_or(&m.name);
            available_base == model_base || m.name == model
        });

        if !model_available {
            let available: Vec<_> = tags.models.iter().map(|m| m.name.as_str()).collect();
            if available.is_empty() {
                return Err(format!(
                    "No models installed. Run 'ollama pull {}'",
                    model
                ));
            }
            return Err(format!(
                "Model '{}' not found. Available: {}. Run 'ollama pull {}'",
                model,
                available.join(", "),
                model
            ));
        }

        Ok(())
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
        let prompt_hash = Self::hash_prompt(&prompt);

        // Check cache first
        if let Some(cached) = self.get_cached(prompt_hash) {
            self.completion_text = Some(cached.clone());
            self.completion_position = Some(cursor_position);
            self.buffer_id = Some(buffer_handle.entity_id());
            cx.notify();
            return;
        }

        let http_client = self.http_client.clone();
        let api_url = self.api_url.clone();
        let model = self.model.clone();
        let buffer_id = buffer_handle.entity_id();
        let should_health_check = !self.health_checked;
        self.health_checked = true;

        self.pending_refresh = Some(cx.spawn(async move |this, cx| {
            if debounce {
                cx.background_executor().timer(DEBOUNCE_TIMEOUT).await;
            }

            // Perform health check on first request
            if should_health_check {
                if let Err(error) = Self::check_health(
                    http_client.clone(),
                    &api_url,
                    &model,
                ).await {
                    log::warn!("Ollama health check failed: {}", error);
                    cx.update(|cx| {
                        set_ollama_connection_status(
                            OllamaConnectionStatus::Error(error),
                            cx,
                        );
                    })?;
                    return Ok(());
                }
            }

            let completion = Self::fetch_completion(
                http_client,
                api_url,
                model,
                prompt,
            ).await;

            match completion {
                Ok(text) => {
                    let text_clone = text.clone();
                    this.update(cx, |this, cx| {
                        // Cache the completion
                        this.cache_completion(prompt_hash, text_clone);
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

    #[test]
    fn test_prompt_hashing() {
        let hash1 = OllamaCompletionProvider::hash_prompt("hello world");
        let hash2 = OllamaCompletionProvider::hash_prompt("hello world");
        let hash3 = OllamaCompletionProvider::hash_prompt("different prompt");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_completion_cache() {
        use http_client::FakeHttpClient;

        let http_client = FakeHttpClient::with_404_response();
        let mut provider = OllamaCompletionProvider::new(http_client);

        // Cache should be empty initially
        assert!(provider.get_cached(12345).is_none());

        // Add a completion
        provider.cache_completion(12345, "completion1".to_string());
        assert_eq!(provider.get_cached(12345), Some(&"completion1".to_string()));

        // Add another
        provider.cache_completion(67890, "completion2".to_string());
        assert_eq!(provider.get_cached(67890), Some(&"completion2".to_string()));
        assert_eq!(provider.get_cached(12345), Some(&"completion1".to_string()));
    }

    #[test]
    fn test_cache_lru_eviction() {
        use http_client::FakeHttpClient;

        let http_client = FakeHttpClient::with_404_response();
        let mut provider = OllamaCompletionProvider::new(http_client);

        // Fill cache beyond capacity
        for i in 0..(MAX_CACHE_SIZE + 10) {
            provider.cache_completion(i as u64, format!("completion{}", i));
        }

        // Cache should be at max size
        assert_eq!(provider.completion_cache.len(), MAX_CACHE_SIZE);

        // First entries should be evicted
        assert!(provider.get_cached(0).is_none());
        assert!(provider.get_cached(9).is_none());

        // Recent entries should still be present
        let last_hash = (MAX_CACHE_SIZE + 9) as u64;
        assert!(provider.get_cached(last_hash).is_some());
    }

    #[test]
    fn test_tags_response_deserialization() {
        let json = r#"{"models": [{"name": "qwen2.5-coder:7b"}, {"name": "codellama:7b"}]}"#;
        let response: TagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.models.len(), 2);
        assert_eq!(response.models[0].name, "qwen2.5-coder:7b");
        assert_eq!(response.models[1].name, "codellama:7b");
    }

    #[test]
    fn test_tags_response_empty() {
        let json = r#"{"models": []}"#;
        let response: TagsResponse = serde_json::from_str(json).unwrap();
        assert!(response.models.is_empty());
    }

    #[test]
    fn test_provider_initial_state() {
        use http_client::FakeHttpClient;

        let http_client = FakeHttpClient::with_404_response();
        let provider = OllamaCompletionProvider::new(http_client);

        assert!(!provider.health_checked);
        assert!(provider.completion_cache.is_empty());
        assert!(provider.cache_order.is_empty());
        assert!(provider.completion_text.is_none());
        assert!(provider.buffer_id.is_none());
    }
}
