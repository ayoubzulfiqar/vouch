//! Egress driver abstraction and shared execution primitives.
//!
//! Drivers translate an [`ExecutionRequest`] into either a complete
//! [`Completion`] or a stream of [`StreamChunk`]s. Concurrency is bounded per
//! backend by a semaphore; [`PermitGuard`] keeps the permit alive for the
//! lifetime of a stream so that slow consumers apply backpressure all the way
//! to the child process or upstream connection.

pub mod attachments;
pub mod cli_pipe;
pub mod http_proxy;
pub mod pty_runner;
pub mod tool_bridge;
pub mod workspace;

use std::{
    collections::BTreeMap,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::{
    config::BackendKind,
    error::{GatewayError, Result},
    metrics, telemetry,
};

/// A single structured content element within a multimodal message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Textual span with optional Anthropic prompt caching metadata.
    Text {
        /// Plain text content.
        text: String,
        /// Optional `cache_control` metadata preserved for compatible upstreams.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    /// Base64-encoded image payload (`image/png`, `image/jpeg`, ...).
    ImageBase64 {
        /// MIME media type.
        media_type: String,
        /// Base64-encoded binary payload (without `data:...;base64,` prefix).
        data: String,
    },
    /// Remote image URL (or non-base64 URI).
    ImageUrl {
        /// Target URL.
        url: String,
    },
}

/// Text or multimodal array message body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageContent {
    /// Plain string message content.
    Text(String),
    /// Structured multimodal content parts.
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl MessageContent {
    /// Flatten text parts into plain text (images omitted for hashing/summaries).
    pub fn as_plain_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text, .. } => {
                            if !text.is_empty() {
                                if !out.is_empty() {
                                    out.push('\n');
                                }
                                out.push_str(text);
                            }
                        }
                        ContentPart::ImageBase64 { .. } => {}
                        ContentPart::ImageUrl { url } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&format!("[image: {url}]"));
                        }
                    }
                }
                out
            }
        }
    }

    /// True when the content contains neither non-empty text nor media parts.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.iter().all(|part| match part {
                ContentPart::Text { text, .. } => text.is_empty(),
                ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => false,
            }),
        }
    }

    /// True if this content carries image parts or `cache_control` metadata.
    pub fn is_rich(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|part| match part {
                ContentPart::Text { cache_control, .. } => cache_control.is_some(),
                ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => true,
            }),
        }
    }

    /// Serialize into OpenAI Chat Completions `"content"` value.
    pub fn to_openai_value(&self) -> Value {
        if !self.is_rich() {
            return Value::String(self.as_plain_text());
        }
        match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Parts(parts) => Value::Array(
                parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text {
                            text,
                            cache_control,
                        } => {
                            let mut obj = serde_json::Map::new();
                            obj.insert("type".into(), json!("text"));
                            obj.insert("text".into(), json!(text));
                            if let Some(cc) = cache_control {
                                obj.insert("cache_control".into(), cc.clone());
                            }
                            Value::Object(obj)
                        }
                        ContentPart::ImageBase64 { media_type, data } => json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{media_type};base64,{data}"),
                            }
                        }),
                        ContentPart::ImageUrl { url } => json!({
                            "type": "image_url",
                            "image_url": {"url": url}
                        }),
                    })
                    .collect(),
            ),
        }
    }
}

/// Normalized JSON/schema structured output request (`response_format` or `text.format`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseFormat {
    /// Valid JSON object/value output without an explicit schema.
    JsonObject,
    /// Strict JSON Schema constrained output.
    JsonSchema {
        /// Optional schema name.
        name: Option<String>,
        /// JSON Schema document.
        schema: Value,
        /// Strict mode indicator.
        strict: bool,
        /// Original raw JSON object preserved for HTTP passthrough.
        raw: Value,
    },
}

impl ResponseFormat {
    /// JSON representation sent upstream to OpenAI-compatible HTTP daemons.
    pub fn to_openai_json(&self) -> Value {
        match self {
            Self::JsonObject => json!({"type": "json_object"}),
            Self::JsonSchema { raw, .. } => raw.clone(),
        }
    }
}

/// Assistant reasoning/thinking trace block (Anthropic extended thinking).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingBlock {
    /// Plain-text thinking trace with optional cryptographic signature.
    Thinking {
        /// Reasoning text.
        thinking: String,
        /// Verification signature, if emitted by upstream.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Opaque redacted thinking payload.
    RedactedThinking {
        /// Encrypted/redacted data string.
        data: String,
    },
}

/// A tool call requested by an assistant response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id (`call_...` or `toolu_...`).
    pub id: String,
    /// Tool / function name.
    pub name: String,
    /// JSON-encoded argument payload.
    pub arguments: String,
}

/// An incremental tool-call delta emitted during streaming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Zero-based tool call index within the turn.
    pub index: usize,
    /// Correlation id, typically emitted on the first chunk for `index`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Function name, typically emitted on the first chunk for `index`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Incremental slice of the JSON arguments string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_delta: Option<String>,
}

/// A single normalized conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// Text or multimodal content parts.
    pub content: MessageContent,
    /// Structured tool calls emitted by an `assistant` turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Correlation id when `role == "tool"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional function name when `role == "tool"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Construct a plain text conversation turn.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Construct a multimodal conversation turn.
    pub fn parts(role: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Parts(parts),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

/// Token accounting returned by a backend, or estimated locally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens.
    pub prompt_tokens: u64,
    /// Output tokens.
    pub completion_tokens: u64,
    /// Sum of input and output.
    pub total_tokens: u64,
}

impl Usage {
    /// Build a usage record and fill the total.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }

    /// BPE token count used when a CLI/PTY backend does not report usage.
    pub fn estimate(prompt: &str, completion: &str) -> Self {
        crate::usage::estimate_usage("", prompt, completion)
    }

    /// Model-aware BPE token count used when a CLI/PTY backend does not report usage.
    pub fn estimate_for_model(model: &str, prompt: &str, completion: &str) -> Self {
        crate::usage::estimate_usage(model, prompt, completion)
    }
}

/// Workspace isolation requested for a single call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceMode {
    /// Use the backend's `isolate_workspace` setting.
    Auto,
    /// Force a fresh/reused worktree for this session.
    Worktree,
    /// Never isolate.
    None,
}

impl WorkspaceMode {
    /// Parse the `X-Workspace-Mode` header.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "worktree" | "isolated" | "isolate" => Some(Self::Worktree),
            "none" | "off" | "disabled" => Some(Self::None),
            _ => None,
        }
    }
}

/// Normalized request passed from ingress to a driver.
#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    /// Correlation id.
    pub request_id: String,
    /// Affinity key (explicit header or deterministic prompt hash).
    pub session_key: String,
    /// Public model name.
    pub model: String,
    /// System prompt, already flattened.
    pub system: Option<String>,
    /// Non-system conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Fully rendered prompt, convenient for CLI drivers.
    pub prompt: String,
    /// Whether the client requested streaming.
    pub stream: bool,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Output token cap.
    pub max_tokens: Option<u32>,
    /// Requested workspace isolation.
    pub workspace_mode: WorkspaceMode,
    /// Working directory assigned by the router (worktree or configured).
    pub workdir: Option<PathBuf>,
    /// Cancellation triggered by client disconnect or shutdown.
    pub cancel: CancellationToken,
    /// Normalized OpenAI-format tool declarations.
    pub tools: Vec<Value>,
    /// Tool selection policy (`"auto"`, `"none"`, `"required"`, or object).
    pub tool_choice: Option<Value>,
    /// Normalized JSON / JSON Schema structured output constraint.
    pub response_format: Option<ResponseFormat>,
    /// Native backend session handle selected for multi-turn CLI continuation.
    pub backend_session_id: Option<String>,
    /// Subset of new turns rendered when resuming via `resume_args`.
    pub delta_messages: Option<Vec<ChatMessage>>,
    /// Slot populated with a newly captured `backend_session_id` after execution.
    pub session_capture: Arc<parking_lot::Mutex<Option<String>>>,
    /// Propagated W3C trace context for child spans and upstream headers.
    pub trace_context: Option<crate::telemetry::TraceContext>,
    /// Selected credential profile id for CLI/PTY rotation (`VOUCH_PROFILE_ID`).
    pub profile_id: Option<String>,
    /// Environment variables overlaid for the active profile.
    pub profile_env: BTreeMap<String, String>,
    /// Optional per-token allowlist restricting eligible backend ids.
    pub allowed_backends: Option<Vec<String>>,
    /// Protocol-specific fields forwarded to HTTP upstreams.
    pub extra: serde_json::Map<String, Value>,
}

impl ExecutionRequest {
    /// True when resuming an existing native CLI session using `resume_args`.
    pub fn is_resuming(&self) -> bool {
        self.delta_messages.is_some()
    }

    /// Conversation messages to render for CLI/PTY (`delta_messages` when resuming,
    /// full `messages` on turn 1).
    pub fn active_messages(&self) -> &[ChatMessage] {
        self.delta_messages.as_deref().unwrap_or(&self.messages)
    }

    /// Render a plain-text prompt for subprocess backends (without temp attachments).
    pub fn rendered_prompt(&self) -> String {
        let mut prompt = if self.is_resuming() || !self.prompt.is_empty() && !self.has_rich_parts()
        {
            if self.is_resuming() {
                render_prompt(None, self.active_messages())
            } else {
                self.prompt.clone()
            }
        } else {
            render_prompt(self.system.as_deref(), &self.messages)
        };
        if !self.tools.is_empty() {
            tool_bridge::append_tool_instructions(
                &mut prompt,
                &self.tools,
                self.tool_choice.as_ref(),
            );
        }
        if let Some(format) = &self.response_format {
            tool_bridge::append_structured_output_instructions(&mut prompt, format);
        }
        prompt
    }

    /// True if any message contains multimodal/rich parts.
    pub fn has_rich_parts(&self) -> bool {
        self.active_messages()
            .iter()
            .any(|message| message.content.is_rich())
    }

    /// OpenAI-shaped messages including the system turn, multimodal parts, and tool calls/results.
    pub fn openai_messages(&self) -> Vec<Value> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if let Some(system) = &self.system
            && !system.is_empty()
        {
            messages.push(json!({"role": "system", "content": system}));
        }
        for message in &self.messages {
            let mut item = serde_json::Map::new();
            item.insert("role".into(), json!(message.role));
            if message.tool_calls.is_empty() {
                item.insert("content".into(), message.content.to_openai_value());
            } else {
                let content = if message.content.is_empty() {
                    Value::Null
                } else {
                    message.content.to_openai_value()
                };
                item.insert("content".into(), content);
                let tool_calls: Vec<Value> = message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        })
                    })
                    .collect();
                item.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            if let Some(call_id) = &message.tool_call_id {
                item.insert("tool_call_id".into(), json!(call_id));
            }
            if let Some(name) = &message.name {
                item.insert("name".into(), json!(name));
            }
            messages.push(Value::Object(item));
        }
        messages
    }
}

/// Render a readable transcript for CLIs that do not speak a protocol.
pub fn render_prompt(system: Option<&str>, messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    if let Some(system) = system
        && !system.trim().is_empty()
    {
        prompt.push_str("System:\n");
        prompt.push_str(system.trim());
        prompt.push_str("\n\n");
    }
    for message in messages {
        tool_bridge::format_message(message, &mut prompt);
    }
    prompt.trim_end().to_string()
}

/// A complete non-streaming result.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Assistant text.
    pub content: String,
    /// Normalized finish reason (`stop`, `tool_calls`, `length`, ...).
    pub finish_reason: String,
    /// Token usage, measured or estimated.
    pub usage: Usage,
    /// Raw upstream JSON when available.
    pub raw: Option<Value>,
    /// Structured tool calls requested by the assistant.
    pub tool_calls: Vec<ToolCall>,
    /// Extended thinking/reasoning blocks emitted by compatible backends.
    pub thinking: Vec<ThinkingBlock>,
}

/// One incremental stream item.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Text appended to the assistant message.
    pub delta: String,
    /// Incremental extended-thinking delta emitted during streaming.
    pub thinking_delta: Option<String>,
    /// Incremental thinking signature delta when streamed.
    pub signature_delta: Option<String>,
    /// Redacted thinking block emitted during streaming.
    pub redacted_thinking: Option<String>,
    /// Incremental tool call deltas.
    pub tool_calls: Vec<ToolCallDelta>,
    /// Set on the terminal chunk.
    pub finish_reason: Option<String>,
    /// Usage, when the backend reports it.
    pub usage: Option<Usage>,
}

impl StreamChunk {
    /// A content delta.
    pub fn delta(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            thinking_delta: None,
            signature_delta: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }

    /// An incremental extended-thinking delta.
    pub fn thinking(delta: impl Into<String>) -> Self {
        Self {
            delta: String::new(),
            thinking_delta: Some(delta.into()),
            signature_delta: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }

    /// A tool-call delta batch.
    pub fn tool_calls(tool_calls: Vec<ToolCallDelta>) -> Self {
        Self {
            delta: String::new(),
            thinking_delta: None,
            signature_delta: None,
            redacted_thinking: None,
            tool_calls,
            finish_reason: None,
            usage: None,
        }
    }

    /// A terminal chunk.
    pub fn finish(reason: impl Into<String>, usage: Option<Usage>) -> Self {
        Self {
            delta: String::new(),
            thinking_delta: None,
            signature_delta: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            finish_reason: Some(reason.into()),
            usage,
        }
    }
}

/// Boxed stream of chunks.
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send + 'static>>;

/// Result of a driver execution.
pub enum DriverResponse {
    /// Fully buffered completion.
    Complete(Completion),
    /// Incremental chunks.
    Stream(ChunkStream),
}

impl std::fmt::Debug for DriverResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Complete(completion) => f
                .debug_struct("Complete")
                .field("chars", &completion.content.chars().count())
                .field("finish_reason", &completion.finish_reason)
                .finish(),
            Self::Stream(_) => f.debug_struct("Stream").finish_non_exhaustive(),
        }
    }
}

/// Normalized request passed to [`Driver::embed`] for `POST /v1/embeddings`.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Correlation id.
    pub request_id: String,
    /// Public model name.
    pub model: String,
    /// Input strings to embed in order.
    pub inputs: Vec<String>,
    /// Requested output encoding (`"float"` or `"base64"`).
    pub encoding_format: Option<String>,
    /// Optional reduced embedding dimension hint.
    pub dimensions: Option<u32>,
    /// Cancellation token.
    pub cancel: CancellationToken,
    /// Propagated W3C trace context.
    pub trace_context: Option<crate::telemetry::TraceContext>,
}

/// Vector embedding output returned by [`Driver::embed`].
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// One embedding vector per input string.
    pub embeddings: Vec<Vec<f32>>,
    /// Measured or estimated BPE token usage.
    pub usage: Usage,
    /// Model name echoed upstream.
    pub model: String,
}

/// Egress transport contract.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Stable backend id.
    fn id(&self) -> &str;
    /// Transport kind.
    fn kind(&self) -> BackendKind;
    /// Configured concurrency limit.
    fn max_concurrency(&self) -> usize;
    /// Execute one request.
    async fn execute(&self, request: ExecutionRequest) -> Result<DriverResponse>;
    /// Generate vector embeddings for `POST /v1/embeddings`.
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResult> {
        let _ = request;
        Err(GatewayError::BadRequest(format!(
            "backend {} does not support embeddings",
            self.id()
        )))
    }
    /// Optional live probe used by `vouch doctor`.
    async fn probe(&self) -> std::result::Result<(), String> {
        Ok(())
    }
}

/// RAII concurrency permit with metrics.
///
/// The permit is an `OwnedSemaphorePermit`, so it can be moved into a spawned
/// task or an async stream. Dropping the guard releases the slot and updates
/// the active-permit gauge.
pub struct PermitGuard {
    permit: Option<OwnedSemaphorePermit>,
    backend: String,
}

impl std::fmt::Debug for PermitGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermitGuard")
            .field("backend", &self.backend)
            .field("held", &self.permit.is_some())
            .finish()
    }
}

impl PermitGuard {
    /// Acquire a permit, waiting at most `wait` and honoring cancellation.
    pub async fn acquire(
        semaphore: Arc<Semaphore>,
        backend: &str,
        wait: Duration,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let started = Instant::now();
        let acquire = semaphore.acquire_owned();
        tokio::pin!(acquire);

        let permit = tokio::select! {
            _ = cancel.cancelled() => return Err(GatewayError::ClientDisconnected),
            result = tokio::time::timeout(wait, &mut acquire) => {
                match result {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) => {
                        return Err(GatewayError::Internal(
                            format!("backend {backend} concurrency semaphore closed"),
                        ));
                    }
                    Err(_) => {
                        return Err(GatewayError::Saturated(format!(
                            "backend {backend} has no free execution slot"
                        )));
                    }
                }
            }
        };

        metrics::record_queue_wait(backend, started.elapsed());
        metrics::inc_active_permit(backend);
        Ok(Self {
            permit: Some(permit),
            backend: backend.to_string(),
        })
    }

    /// Release the permit immediately.
    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if self.permit.take().is_some() {
            metrics::dec_active_permit(&self.backend);
        }
    }
}

impl Drop for PermitGuard {
    fn drop(&mut self) {
        self.release_inner();
    }
}

// ---------------------------------------------------------------------------
// Process-group supervision
// ---------------------------------------------------------------------------

/// RAII supervisor for a Unix process group.
///
/// Every subprocess is spawned with `.process_group(0)`, so its PGID equals
/// its PID. Dropping the guard (request timeout, driver error, SSE client
/// disconnect, or normal task cancellation) sends `SIGTERM` to the entire
/// group, waits 500 ms on a dedicated thread, then sends `SIGKILL` if any
/// member remains. This guarantees that grandchildren do not outlive a request.
#[cfg(unix)]
#[derive(Debug)]
pub struct ProcessGroupGuard {
    inner: Arc<GroupKillSwitch>,
}

#[cfg(unix)]
#[derive(Debug)]
struct GroupKillSwitch {
    pgid: i32,
    backend: String,
    armed: std::sync::atomic::AtomicBool,
}

#[cfg(unix)]
impl ProcessGroupGuard {
    /// Create an armed guard for `pgid`.
    pub fn new(pgid: u32, backend: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(GroupKillSwitch {
                pgid: pgid as i32,
                backend: backend.into(),
                armed: std::sync::atomic::AtomicBool::new(true),
            }),
        }
    }

    /// Share the same kill switch with another task.
    pub fn clone_guard(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Mark the process group as successfully finished; dropping will not
    /// signal it.
    pub fn disarm(&self) {
        self.inner
            .armed
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Send `SIGTERM` now and schedule `SIGKILL` after the grace period.
    ///
    /// Idempotent: only the first caller performs the escalation.
    pub fn terminate(&self) {
        use std::sync::atomic::Ordering;

        if !self.inner.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        let pgid = self.inner.pgid;
        let backend = self.inner.backend.clone();
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(pgid),
            nix::sys::signal::Signal::SIGTERM,
        );
        let _ = std::thread::Builder::new()
            .name(format!("vouch-kill-{backend}"))
            .spawn(move || {
                std::thread::sleep(Duration::from_millis(500));
                // Only escalate if the group still exists. This avoids
                // signalling a recycled PGID after a clean exit.
                if nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pgid), None).is_ok() {
                    let _ = nix::sys::signal::killpg(
                        nix::unistd::Pid::from_raw(pgid),
                        nix::sys::signal::Signal::SIGKILL,
                    );
                }
            });
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Windows Job Object supervision ensuring all descendant processes exit when terminated or dropped.
#[cfg(windows)]
#[derive(Debug)]
pub struct ProcessGroupGuard {
    inner: Arc<WindowsJobSwitch>,
}

#[cfg(windows)]
#[derive(Debug)]
struct WindowsJobSwitch {
    job_handle: isize,
    backend: String,
    armed: std::sync::atomic::AtomicBool,
}

#[cfg(windows)]
impl WindowsJobSwitch {
    fn close_handle(&self) {
        if self.job_handle != 0 {
            // Safety: `job_handle` is a valid Win32 Job Object handle owned exclusively by this struct.
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(
                    self.job_handle as windows_sys::Win32::Foundation::HANDLE,
                );
            }
        }
    }
}

#[cfg(windows)]
impl Drop for WindowsJobSwitch {
    fn drop(&mut self) {
        self.close_handle();
    }
}

#[cfg(windows)]
impl ProcessGroupGuard {
    /// Create a Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and attach `pid`.
    pub fn new(pid: u32, backend: impl Into<String>) -> Self {
        use std::mem::{size_of, zeroed};
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_ALL_ACCESS};

        let backend = backend.into();
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            tracing::warn!(backend = %backend, "CreateJobObjectW failed; child tree supervision disabled");
            return Self {
                inner: Arc::new(WindowsJobSwitch {
                    job_handle: 0,
                    backend,
                    armed: std::sync::atomic::AtomicBool::new(false),
                }),
            };
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            tracing::warn!(backend = %backend, "SetInformationJobObject(KILL_ON_JOB_CLOSE) failed");
        }
        let proc_handle = unsafe { OpenProcess(PROCESS_ALL_ACCESS, 0, pid) };
        if !proc_handle.is_null() {
            let assigned = unsafe { AssignProcessToJobObject(job, proc_handle) };
            if assigned == 0 {
                tracing::warn!(backend = %backend, pid, "AssignProcessToJobObject failed");
            }
            unsafe { CloseHandle(proc_handle) };
        } else {
            tracing::warn!(backend = %backend, pid, "OpenProcess failed for Job Object attachment");
        }
        Self {
            inner: Arc::new(WindowsJobSwitch {
                job_handle: job as isize,
                backend,
                armed: std::sync::atomic::AtomicBool::new(true),
            }),
        }
    }

    /// Share the same Job Object kill switch with another task.
    pub fn clone_guard(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Disarm termination when the child completed cleanly.
    pub fn disarm(&self) {
        self.inner
            .armed
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Terminate the entire Windows Job Object process tree immediately.
    pub fn terminate(&self) {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        if !self.inner.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        if self.inner.job_handle != 0 {
            let ok = unsafe {
                TerminateJobObject(
                    self.inner.job_handle as windows_sys::Win32::Foundation::HANDLE,
                    1,
                )
            };
            if ok == 0 {
                tracing::warn!(backend = %self.inner.backend, "TerminateJobObject returned failure");
            }
        }
    }
}

#[cfg(windows)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Non-Unix, non-Windows fallback stub with the same API.
#[cfg(not(any(unix, windows)))]
#[derive(Debug, Clone)]
pub struct ProcessGroupGuard {
    backend: String,
}

#[cfg(not(any(unix, windows)))]
impl ProcessGroupGuard {
    /// Create a stub guard.
    pub fn new(_pgid: u32, backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
        }
    }

    /// Stub.
    pub fn clone_guard(&self) -> Self {
        self.clone()
    }

    /// Stub.
    pub fn disarm(&self) {}

    /// Stub.
    pub fn terminate(&self) {
        tracing::warn!(
            backend = %self.backend,
            "process-group supervision is not supported on this platform"
        );
    }
}

// ---------------------------------------------------------------------------
// ANSI stripping
// ---------------------------------------------------------------------------

struct StripperPerformer<'a> {
    output: &'a mut String,
}

impl vte::Perform for StripperPerformer<'_> {
    fn print(&mut self, c: char) {
        self.output.push(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\n' | b'\t' => self.output.push(byte as char),
            // Drop carriage returns: PTYs emit CRLF, API clients expect LF.
            _ => {}
        }
    }
}

/// Stateful ANSI/VT escape-sequence stripper.
///
/// A single instance must be reused across chunks so that escape sequences
/// split across reads are removed correctly.
pub struct AnsiStripper {
    parser: vte::Parser,
    output: String,
}

impl AnsiStripper {
    /// Create a stripper.
    pub fn new() -> Self {
        Self {
            parser: vte::Parser::new(),
            output: String::new(),
        }
    }

    /// Feed bytes and return the printable text produced by this chunk.
    pub fn feed(&mut self, bytes: &[u8]) -> String {
        let Self { parser, output } = self;
        output.clear();
        let mut performer = StripperPerformer { output };
        parser.advance(&mut performer, bytes);
        std::mem::take(output)
    }
}

impl Default for AnsiStripper {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip ANSI escapes from a complete byte buffer.
pub fn strip_ansi(bytes: &[u8]) -> String {
    let mut stripper = AnsiStripper::new();
    stripper.feed(bytes)
}

/// Stateful UTF-8 boundary decoder for incremental subprocess `read` loops.
///
/// Decodes complete leading UTF-8 codepoints immediately via `std::str::from_utf8`
/// (`valid_up_to()`), replaces broken mid-stream byte sequences with `U+FFFD`,
/// and carries incomplete trailing multi-byte UTF-8 sequences over to the next
/// read call so characters split across OS pipe buffers are never corrupted.
#[derive(Debug, Default)]
pub struct Utf8ChunkDecoder {
    pending: Vec<u8>,
}

impl Utf8ChunkDecoder {
    /// Create a decoder with an empty carry buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes and return the valid UTF-8 prefix string ready for processing.
    pub fn feed(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0
                        && let Ok(valid) = std::str::from_utf8(&self.pending[..valid_up_to])
                    {
                        out.push_str(valid);
                    }
                    match error.error_len() {
                        Some(bad_len) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid_up_to + bad_len);
                        }
                        None => {
                            if valid_up_to > 0 {
                                self.pending.drain(..valid_up_to);
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Flush any incomplete trailing bytes at EOF cleanly as replacement characters.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let remainder = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        remainder
    }
}

// ---------------------------------------------------------------------------
// SSE decoding for upstream streams
// ---------------------------------------------------------------------------

/// One decoded server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Optional `event:` field.
    pub event: Option<String>,
    /// Joined `data:` fields.
    pub data: String,
}

/// Incremental SSE decoder that tolerates arbitrary chunk boundaries.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseDecoder {
    /// Create a decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes and return every complete event.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=position).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line).into_owned();
            self.line(&line, &mut events);
        }
        events
    }

    /// Flush a final event that was not terminated by a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer).into_owned();
            self.buffer.clear();
            let mut events = Vec::new();
            self.line(&line, &mut events);
        }
        self.emit()
    }

    fn line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if let Some(event) = self.emit() {
                events.push(event);
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            _ => {}
        }
    }

    fn emit(&mut self) -> Option<SseEvent> {
        if self.data.is_empty() {
            self.event = None;
            return None;
        }
        let data = self.data.join("\n");
        self.data.clear();
        let event = self.event.take();
        Some(SseEvent { event, data })
    }
}

// ---------------------------------------------------------------------------
// Upstream JSON helpers
// ---------------------------------------------------------------------------

/// Extract assistant text from an OpenAI, Anthropic, or generic JSON payload.
pub fn extract_content(value: &Value) -> Option<String> {
    if value.get("error").is_some() {
        return None;
    }
    if let Some(choices) = value.get("choices").and_then(Value::as_array)
        && let Some(choice) = choices.first()
    {
        if let Some(content) = choice
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(content_to_text)
        {
            return Some(content);
        }
        if let Some(content) = choice
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(content_to_text)
        {
            return Some(content);
        }
        if let Some(text) = choice.get("text").and_then(Value::as_str) {
            return Some(text.to_string());
        }
    }
    for pointer in [
        "/content",
        "/text",
        "/response",
        "/output",
        "/result",
        "/message/content",
    ] {
        if let Some(content) = value.pointer(pointer).and_then(content_to_text) {
            return Some(content);
        }
    }
    value.as_str().map(ToString::to_string)
}

/// Extract tool calls from an OpenAI or Anthropic complete JSON payload.
pub fn extract_tool_calls(value: &Value) -> Vec<ToolCall> {
    if let Some(calls) = value
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
    {
        return calls
            .iter()
            .filter_map(|call| {
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                let function = call.get("function").unwrap_or(call);
                let name = function.get("name").and_then(Value::as_str)?.to_string();
                let arguments = match function.get("arguments") {
                    Some(Value::String(raw)) => raw.clone(),
                    Some(other) => other.to_string(),
                    None => "{}".to_string(),
                };
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();
    }
    if let Some(blocks) = value.get("content").and_then(Value::as_array) {
        return blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .filter_map(|block| {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                let name = block.get("name").and_then(Value::as_str)?.to_string();
                let arguments = block
                    .get("input")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "{}".to_string());
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();
    }
    Vec::new()
}

/// Extract incremental tool-call deltas from one OpenAI SSE event payload.
pub fn extract_tool_call_deltas(value: &Value) -> Vec<ToolCallDelta> {
    let Some(calls) = value
        .pointer("/choices/0/delta/tool_calls")
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    calls
        .iter()
        .enumerate()
        .map(|(fallback_idx, item)| {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(fallback_idx);
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let function = item.get("function").unwrap_or(item);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let arguments_delta = function
                .get("arguments")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            }
        })
        .collect()
}

/// Extract extended-thinking / redacted-thinking blocks from an upstream JSON response.
pub fn extract_thinking(value: &Value) -> Vec<ThinkingBlock> {
    let mut blocks = Vec::new();
    if let Some(parts) = value.get("content").and_then(Value::as_array) {
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("thinking") => {
                    if let Some(thinking) = part.get("thinking").and_then(Value::as_str) {
                        blocks.push(ThinkingBlock::Thinking {
                            thinking: thinking.to_string(),
                            signature: part
                                .get("signature")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                        });
                    }
                }
                Some("redacted_thinking") => {
                    if let Some(data) = part.get("data").and_then(Value::as_str) {
                        blocks.push(ThinkingBlock::RedactedThinking {
                            data: data.to_string(),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    if blocks.is_empty()
        && let Some(reasoning) = value
            .pointer("/choices/0/message/reasoning_content")
            .or_else(|| value.pointer("/choices/0/message/reasoning"))
            .and_then(content_to_text)
        && !reasoning.is_empty()
    {
        blocks.push(ThinkingBlock::Thinking {
            thinking: reasoning,
            signature: None,
        });
    }
    blocks
}

/// Detailed incremental delta extracted from an upstream SSE event.
#[derive(Debug, Default)]
pub struct ExtractedDelta {
    /// Textual assistant content delta.
    pub content: Option<String>,
    /// Extended thinking delta.
    pub thinking: Option<String>,
    /// Thinking cryptographic signature delta.
    pub signature: Option<String>,
    /// Redacted thinking payload.
    pub redacted_thinking: Option<String>,
    /// Finish reason, if present.
    pub finish_reason: Option<String>,
    /// Usage metrics, if present.
    pub usage: Option<Usage>,
}

/// Extract content, thinking deltas, finish reason, and usage from an upstream event.
pub fn extract_delta_full(value: &Value) -> ExtractedDelta {
    let usage = extract_usage(value);
    if let Some(choices) = value.get("choices").and_then(Value::as_array)
        && let Some(choice) = choices.first()
    {
        let content = choice
            .get("delta")
            .and_then(|delta| delta.get("content").and_then(content_to_text))
            .or_else(|| {
                choice
                    .get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            });
        let thinking = choice.get("delta").and_then(|delta| {
            delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(content_to_text)
        });
        let finish_reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        if content.is_some() || thinking.is_some() || finish_reason.is_some() {
            return ExtractedDelta {
                content,
                thinking,
                finish_reason,
                usage,
                ..Default::default()
            };
        }
    }

    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match event_type {
        "content_block_start" => {
            if let Some(cb) = value.get("content_block")
                && cb.get("type").and_then(Value::as_str) == Some("redacted_thinking")
                && let Some(data) = cb.get("data").and_then(Value::as_str)
            {
                return ExtractedDelta {
                    redacted_thinking: Some(data.to_string()),
                    usage,
                    ..Default::default()
                };
            }
            return ExtractedDelta {
                usage,
                ..Default::default()
            };
        }
        "content_block_delta" => {
            if let Some(delta_obj) = value.get("delta") {
                match delta_obj.get("type").and_then(Value::as_str) {
                    Some("thinking_delta") => {
                        return ExtractedDelta {
                            thinking: delta_obj
                                .get("thinking")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            usage,
                            ..Default::default()
                        };
                    }
                    Some("signature_delta") => {
                        return ExtractedDelta {
                            signature: delta_obj
                                .get("signature")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            usage,
                            ..Default::default()
                        };
                    }
                    _ => {
                        return ExtractedDelta {
                            content: delta_obj
                                .get("text")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            usage,
                            ..Default::default()
                        };
                    }
                }
            }
        }
        "message_delta" => {
            let finish_reason = value
                .get("delta")
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str)
                .map(ToString::to_string);
            return ExtractedDelta {
                finish_reason,
                usage,
                ..Default::default()
            };
        }
        "message_stop" | "done" => {
            return ExtractedDelta {
                finish_reason: Some("stop".into()),
                usage,
                ..Default::default()
            };
        }
        "error" => {
            return ExtractedDelta {
                usage,
                ..Default::default()
            };
        }
        _ => {}
    }

    if let Some(delta) = value.get("delta") {
        if let Some(text) = delta.as_str() {
            return ExtractedDelta {
                content: Some(text.to_string()),
                usage,
                ..Default::default()
            };
        }
        if let Some(text) = delta
            .get("text")
            .and_then(Value::as_str)
            .or_else(|| delta.get("content").and_then(Value::as_str))
        {
            return ExtractedDelta {
                content: Some(text.to_string()),
                usage,
                ..Default::default()
            };
        }
    }
    for key in ["content", "text", "token", "response"] {
        if let Some(text) = value.get(key).and_then(Value::as_str) {
            return ExtractedDelta {
                content: Some(text.to_string()),
                usage,
                ..Default::default()
            };
        }
    }
    ExtractedDelta {
        usage,
        ..Default::default()
    }
}

/// Extract a stream delta, finish reason, and usage from one upstream event.
pub fn extract_delta(value: &Value) -> (Option<String>, Option<String>, Option<Usage>) {
    let extracted = extract_delta_full(value);
    (
        extracted.content.or(extracted.thinking),
        extracted.finish_reason,
        extracted.usage,
    )
}

/// Extract token usage from common upstream shapes.
pub fn extract_usage(value: &Value) -> Option<Usage> {
    let usage = value.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if prompt == 0 && completion == 0 {
        return None;
    }
    Some(Usage::new(prompt, completion))
}

/// Extract a human-readable upstream error message.
pub fn extract_upstream_error(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    if let Some(message) = error.get("message").and_then(Value::as_str) {
        return Some(message.to_string());
    }
    if let Some(message) = error.as_str() {
        return Some(message.to_string());
    }
    Some(error.to_string())
}

/// Convert any JSON value into text, flattening content-part arrays.
pub fn content_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut output = String::new();
            for part in parts {
                if matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("thinking" | "redacted_thinking")
                ) {
                    continue;
                }
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("content").and_then(Value::as_str))
                    .or_else(|| part.as_str());
                if let Some(text) = text {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(text);
                } else if part.get("type").and_then(Value::as_str) == Some("image") {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str("[image omitted]");
                }
            }
            if output.is_empty() {
                None
            } else {
                Some(output)
            }
        }
        Value::Object(_) => value
            .get("text")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// Map an upstream HTTP status and headers to a gateway error, redacting the body.
pub fn map_upstream_status_with_headers(
    status: u16,
    headers: Option<&reqwest::header::HeaderMap>,
    body: &str,
) -> GatewayError {
    let body = telemetry::redact_text(body);
    let body = truncate(&body, 800);
    match status {
        429 => {
            let retry_after = headers
                .and_then(|map| map.get(reqwest::header::RETRY_AFTER))
                .and_then(|val| val.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            GatewayError::upstream_rate_limited(
                format!("upstream returned 429: {body}"),
                retry_after,
            )
        }
        400 | 422 => GatewayError::BadRequest(format!("upstream rejected request: {body}")),
        401 | 403 => GatewayError::UpstreamAuthExpired(format!(
            "upstream authentication failed ({status}): {body}"
        )),
        404 => GatewayError::NotFound(format!("upstream resource not found: {body}")),
        _ => GatewayError::Upstream(format!("upstream returned HTTP {status}: {body}")),
    }
}

/// Map an upstream HTTP status to a gateway error, redacting the body.
pub fn map_upstream_status(status: u16, body: &str) -> GatewayError {
    map_upstream_status_with_headers(status, None, body)
}

/// Classify a CLI/PTY failure (stderr/message) into auth-expiry, rate-limit,
/// or a generic upstream failure.
pub fn classify_cli_failure(backend: &str, exit_detail: &str, stderr: &str) -> GatewayError {
    let combined = format!("{stderr}\n{exit_detail}").to_ascii_lowercase();
    let snippet = truncate(&telemetry::redact_text(stderr.trim()), 500);
    let detail_line = if snippet.is_empty() {
        format!("backend {backend} {exit_detail}")
    } else {
        format!("backend {backend} {exit_detail}: {snippet}")
    };

    const AUTH_MARKERS: &[&str] = &[
        "invalid token",
        "token expired",
        "expired token",
        "please sign in",
        "sign in required",
        "not logged in",
        "unauthorized",
        "authentication failed",
        "invalid api key",
        "invalid_api_key",
        "401 unauthorized",
        "403 forbidden",
    ];
    if AUTH_MARKERS.iter().any(|marker| combined.contains(marker)) {
        return GatewayError::UpstreamAuthExpired(detail_line);
    }

    const RATE_MARKERS: &[&str] = &[
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota exceeded",
        "429",
    ];
    if RATE_MARKERS.iter().any(|marker| combined.contains(marker)) {
        return GatewayError::upstream_rate_limited(detail_line, None);
    }

    GatewayError::Upstream(detail_line)
}

/// Buffer stream chunks until the first non-whitespace delta, tool call, or
/// thinking trace arrives. If an error occurs while only empty/whitespace
/// output was produced, returns `Err` so [`crate::router::Router`] can fail
/// over transparently to the next backend.
pub async fn prime_stream(mut stream: ChunkStream) -> Result<ChunkStream> {
    let mut buffered = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                let meaningful = chunk.delta.chars().any(|c| !c.is_whitespace())
                    || chunk
                        .thinking_delta
                        .as_deref()
                        .is_some_and(|t| t.chars().any(|c| !c.is_whitespace()))
                    || chunk.redacted_thinking.is_some()
                    || !chunk.tool_calls.is_empty()
                    || chunk.finish_reason.is_some();
                buffered.push(Ok(chunk));
                if meaningful {
                    let prefix = futures::stream::iter(buffered);
                    return Ok(Box::pin(prefix.chain(stream)));
                }
            }
            Err(error) => {
                return Err(error);
            }
        }
    }
    Ok(Box::pin(futures::stream::iter(buffered)))
}

/// Capture an upstream native session handle using `session_id_regex`.
pub(crate) fn capture_session_id(
    regex_str: Option<&str>,
    stderr: &str,
    stdout: &str,
    capture_slot: &parking_lot::Mutex<Option<String>>,
) {
    let Some(pattern) = regex_str else {
        return;
    };
    let Ok(regex) = regex::Regex::new(pattern) else {
        return;
    };
    let candidate = [stderr, stdout].into_iter().find_map(|haystack| {
        regex.captures(haystack).and_then(|caps| {
            caps.get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().trim().to_string())
        })
    });
    if let Some(id) = candidate
        && !id.is_empty()
    {
        *capture_slot.lock() = Some(id);
    }
}

/// Truncate text to at most `limit` characters.
pub fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut output: String = text.chars().take(limit).collect();
    output.push('…');
    output
}

/// Aggregated result from collecting a [`ChunkStream`].
#[derive(Debug, Clone)]
pub struct CollectedStream {
    /// Accumulated assistant text.
    pub content: String,
    /// Reassembled structured tool calls.
    pub tool_calls: Vec<ToolCall>,
    /// Accumulated extended thinking blocks.
    pub thinking: Vec<ThinkingBlock>,
    /// Terminal finish reason, if emitted.
    pub finish_reason: Option<String>,
    /// Token usage, if emitted.
    pub usage: Option<Usage>,
}

/// Collect a stream into a bounded buffer, assembling text and tool calls.
pub async fn collect_stream_full(
    mut stream: ChunkStream,
    max_bytes: usize,
) -> Result<CollectedStream> {
    let mut content = String::new();
    let mut thinking_text = String::new();
    let mut thinking_sig: Option<String> = None;
    let mut thinking_blocks = Vec::new();
    let mut finish = None;
    let mut usage = None;
    let mut tool_calls: std::collections::BTreeMap<usize, ToolCall> =
        std::collections::BTreeMap::new();
    let mut total_bytes = 0usize;

    while let Some(item) = stream.next().await {
        let chunk = item?;
        total_bytes = total_bytes.saturating_add(chunk.delta.len());
        content.push_str(&chunk.delta);
        if let Some(t) = chunk.thinking_delta {
            total_bytes = total_bytes.saturating_add(t.len());
            thinking_text.push_str(&t);
        }
        if let Some(sig) = chunk.signature_delta {
            thinking_sig.get_or_insert_with(String::new).push_str(&sig);
        }
        if let Some(data) = chunk.redacted_thinking {
            if !thinking_text.is_empty() {
                thinking_blocks.push(ThinkingBlock::Thinking {
                    thinking: std::mem::take(&mut thinking_text),
                    signature: thinking_sig.take(),
                });
            }
            thinking_blocks.push(ThinkingBlock::RedactedThinking { data });
        }
        for delta in chunk.tool_calls {
            let entry = tool_calls.entry(delta.index).or_insert_with(|| ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                name: String::new(),
                arguments: String::new(),
            });
            if let Some(id) = delta.id {
                entry.id = id;
            }
            if let Some(name) = delta.name {
                entry.name.push_str(&name);
            }
            if let Some(args) = delta.arguments_delta {
                total_bytes = total_bytes.saturating_add(args.len());
                entry.arguments.push_str(&args);
            }
        }
        if total_bytes > max_bytes {
            return Err(GatewayError::Internal(
                "backend output exceeded the configured safety limit".into(),
            ));
        }
        if chunk.finish_reason.is_some() {
            finish = chunk.finish_reason;
        }
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
    }
    if !thinking_text.is_empty() {
        thinking_blocks.push(ThinkingBlock::Thinking {
            thinking: thinking_text,
            signature: thinking_sig,
        });
    }

    Ok(CollectedStream {
        content,
        tool_calls: tool_calls.into_values().collect(),
        thinking: thinking_blocks,
        finish_reason: finish,
        usage,
    })
}

/// Collect a stream into a bounded buffer.
pub async fn collect_stream(
    stream: ChunkStream,
    max_bytes: usize,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let collected = collect_stream_full(stream, max_bytes).await?;
    Ok((collected.content, collected.finish_reason, collected.usage))
}

/// Finalize a complete CLI/PTY subprocess output into a [`Completion`].
pub(crate) fn finalize_subprocess_completion(
    model: &str,
    prompt: &str,
    raw_text: &str,
    output_mode: crate::config::OutputMode,
    parse_tools: bool,
    response_format: Option<&ResponseFormat>,
) -> Result<Completion> {
    let (mut content, tool_calls) = match output_mode {
        crate::config::OutputMode::Text if parse_tools => {
            tool_bridge::parse_tool_calls(raw_text.trim())
        }
        crate::config::OutputMode::Text => (raw_text.trim().to_string(), Vec::new()),
        crate::config::OutputMode::Json => (
            crate::driver::cli_pipe::parse_json_content(raw_text)?,
            Vec::new(),
        ),
        crate::config::OutputMode::JsonLines => (
            crate::driver::cli_pipe::parse_json_lines(raw_text),
            Vec::new(),
        ),
    };
    if tool_calls.is_empty()
        && let Some(format) = response_format
    {
        content = tool_bridge::enforce_response_format(&content, format)?;
    }
    let finish_reason = if tool_calls.is_empty() {
        "stop".to_string()
    } else {
        "tool_calls".to_string()
    };
    let usage = Usage::estimate_for_model(model, prompt, raw_text);
    Ok(Completion {
        content,
        finish_reason,
        usage,
        raw: None,
        tool_calls,
        thinking: Vec::new(),
    })
}

/// Emit buffered tail chunks (and optional parsed tool calls / validated JSON)
/// at subprocess EOF.
pub(crate) async fn emit_subprocess_tail(
    sender: &tokio::sync::mpsc::Sender<Result<StreamChunk>>,
    model: &str,
    prompt: &str,
    produced: &str,
    output_mode: crate::config::OutputMode,
    parse_tools: bool,
    response_format: Option<&ResponseFormat>,
) -> bool {
    if output_mode == crate::config::OutputMode::Text && (parse_tools || response_format.is_some())
    {
        let (mut clean_text, tool_calls) = if parse_tools {
            tool_bridge::parse_tool_calls(produced)
        } else {
            (produced.trim().to_string(), Vec::new())
        };
        if tool_calls.is_empty()
            && let Some(format) = response_format
        {
            clean_text = match tool_bridge::enforce_response_format(&clean_text, format) {
                Ok(valid) => valid,
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                    return false;
                }
            };
        }
        if !clean_text.is_empty()
            && sender
                .send(Ok(StreamChunk::delta(clean_text)))
                .await
                .is_err()
        {
            return false;
        }
        if !tool_calls.is_empty() {
            let deltas = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| ToolCallDelta {
                    index,
                    id: Some(call.id),
                    name: Some(call.name),
                    arguments_delta: Some(call.arguments),
                })
                .collect();
            if sender
                .send(Ok(StreamChunk::tool_calls(deltas)))
                .await
                .is_err()
            {
                return false;
            }
            let usage = Usage::estimate_for_model(model, prompt, produced);
            return sender
                .send(Ok(StreamChunk::finish("tool_calls", Some(usage))))
                .await
                .is_ok();
        }
    }
    let usage = Usage::estimate_for_model(model, prompt, produced);
    sender
        .send(Ok(StreamChunk::finish("stop", Some(usage))))
        .await
        .is_ok()
}

/// Send pre-built stream chunks over the bounded channel, returning `false` if
/// the downstream consumer has disconnected.
pub(crate) async fn dispatch_stream_chunks(
    sender: &tokio::sync::mpsc::Sender<Result<StreamChunk>>,
    chunks: Vec<StreamChunk>,
) -> bool {
    for chunk in chunks {
        if sender.send(Ok(chunk)).await.is_err() {
            return false;
        }
    }
    true
}

/// Flush remaining text/tool-call buffers for `OutputMode::Text` streams,
/// enforce any `response_format`, and send the terminal finish chunk.
pub(crate) async fn finish_subprocess_text_stream(
    sender: &tokio::sync::mpsc::Sender<Result<StreamChunk>>,
    filter: &mut tool_bridge::SubprocessStreamFilter,
    model: &str,
    prompt: &str,
    produced: &str,
) -> bool {
    match filter.finish() {
        Ok((tail_chunks, reason)) => {
            if !dispatch_stream_chunks(sender, tail_chunks).await {
                return false;
            }
            let usage = Usage::estimate_for_model(model, prompt, produced);
            sender
                .send(Ok(StreamChunk::finish(reason, Some(usage))))
                .await
                .is_ok()
        }
        Err(error) => {
            let _ = sender.send(Err(error)).await;
            false
        }
    }
}

/// Compiled `[[backends.output_filters]]` regex rules for CLI/PTY output.
#[derive(Debug, Clone, Default)]
pub struct OutputFilterSet {
    rules: Vec<(regex::Regex, String)>,
}

impl OutputFilterSet {
    /// Compile configured output filter patterns.
    pub fn compile(filters: &[crate::config::OutputFilterConfig]) -> Result<Self> {
        let mut rules = Vec::with_capacity(filters.len());
        for filter in filters {
            let regex = regex::Regex::new(&filter.pattern).map_err(|error| {
                GatewayError::Config(format!(
                    "invalid output_filters regex {:?}: {error}",
                    filter.pattern
                ))
            })?;
            rules.push((regex, filter.replace.clone()));
        }
        Ok(Self { rules })
    }

    /// True when no output filters are configured.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Apply every configured regex replacement sequentially.
    pub fn apply(&self, input: &str) -> String {
        if self.rules.is_empty() || input.is_empty() {
            return input.to_string();
        }
        let mut out = input.to_string();
        for (regex, replace) in &self.rules {
            out = regex.replace_all(&out, replace.as_str()).into_owned();
        }
        out
    }
}

/// Incremental text filter adapter for streaming CLI/PTY output.
///
/// Passes text through unbuffered when `filters` is empty; otherwise buffers
/// to newline boundaries (and flushes on EOF) so line/inline regex rules match
/// without splitting across pipe/pty read chunks.
pub(crate) struct StreamOutputFilterAdapter {
    filters: OutputFilterSet,
    carry: String,
}

impl StreamOutputFilterAdapter {
    pub(crate) fn new(filters: OutputFilterSet) -> Self {
        Self {
            filters,
            carry: String::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> String {
        if self.filters.is_empty() {
            return text.to_string();
        }
        self.carry.push_str(text);
        let Some(last_nl) = self.carry.rfind('\n') else {
            return String::new();
        };
        let ready: String = self.carry.drain(..=last_nl).collect();
        self.filters.apply(&ready)
    }

    pub(crate) fn finish(&mut self) -> String {
        if self.filters.is_empty() || self.carry.is_empty() {
            return String::new();
        }
        let remainder = std::mem::take(&mut self.carry);
        self.filters.apply(&remainder)
    }

    pub(crate) fn apply_raw(&self, text: &str) -> String {
        self.filters.apply(text)
    }
}

/// Convert a byte-count-limited error into a readable message.
pub fn io_error(context: &str, error: std::io::Error) -> GatewayError {
    GatewayError::Io(std::io::Error::new(
        error.kind(),
        format!("{context}: {error}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_stripper_removes_color_and_cursor_codes() {
        let mut stripper = AnsiStripper::new();
        let output = stripper.feed(b"\x1b[31mred\x1b[0m \x1b[2Ktext\r\n");
        assert_eq!(output, "red text\n");
    }

    #[test]
    fn ansi_stripper_handles_split_sequences() {
        let mut stripper = AnsiStripper::new();
        let first = stripper.feed(b"hello \x1b[3");
        let second = stripper.feed(b"1mworld\x1b[0m");
        assert_eq!(first, "hello ");
        assert_eq!(second, "world");
    }

    #[test]
    fn sse_decoder_handles_multiline_events() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"event: message_start\ndata: {\"a\":1}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{\"a\":1}");
    }

    #[test]
    fn sse_decoder_handles_split_chunks() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.push(b"data: {\"x\":").is_empty());
        let events = decoder.push(b"1}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"x\":1}");
    }

    #[test]
    fn extracts_openai_and_anthropic_content() {
        let openai = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}]
        });
        assert_eq!(extract_content(&openai).as_deref(), Some("hello"));
        let anthropic = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        });
        assert_eq!(extract_content(&anthropic).as_deref(), Some("hi\nthere"));
    }

    #[test]
    fn parses_workspace_header() {
        assert_eq!(
            WorkspaceMode::parse("worktree"),
            Some(WorkspaceMode::Worktree)
        );
        assert_eq!(WorkspaceMode::parse("none"), Some(WorkspaceMode::None));
        assert_eq!(WorkspaceMode::parse("bogus"), None);
    }

    #[test]
    fn utf8_chunk_decoder_handles_split_multibyte_codepoints() {
        let mut decoder = Utf8ChunkDecoder::new();
        let crab = "hi 🦀!".as_bytes();
        // Split right inside the 4-byte '🦀' sequence
        assert_eq!(decoder.feed(&crab[..4]), "hi ");
        assert_eq!(decoder.feed(&crab[4..5]), "");
        assert_eq!(decoder.feed(&crab[5..]), "🦀!");
        assert_eq!(decoder.finish(), "");
    }
}
