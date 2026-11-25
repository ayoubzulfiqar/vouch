//! Egress driver abstraction and shared execution primitives.
//!
//! Drivers translate an [`ExecutionRequest`] into either a complete
//! [`Completion`] or a stream of [`StreamChunk`]s. Concurrency is bounded per
//! backend by a semaphore; [`PermitGuard`] keeps the permit alive for the
//! lifetime of a stream so that slow consumers apply backpressure all the way
//! to the child process or upstream connection.

pub mod cli_pipe;
pub mod http_proxy;
pub mod pty_runner;
pub mod workspace;

use std::{
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::{
    config::BackendKind,
    error::{GatewayError, Result},
    metrics, telemetry,
};

/// A single normalized conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Flattened textual content.
    pub content: String,
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

    /// Coarse estimate used when a CLI backend does not report usage.
    pub fn estimate(prompt: &str, completion: &str) -> Self {
        let prompt_tokens = (prompt.chars().count() as u64).div_ceil(4);
        let completion_tokens = (completion.chars().count() as u64).div_ceil(4);
        Self::new(prompt_tokens, completion_tokens)
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
    /// Protocol-specific fields forwarded to HTTP upstreams.
    pub extra: serde_json::Map<String, Value>,
}

impl ExecutionRequest {
    /// Render a plain-text prompt for subprocess backends.
    pub fn rendered_prompt(&self) -> String {
        if !self.prompt.is_empty() {
            return self.prompt.clone();
        }
        render_prompt(self.system.as_deref(), &self.messages)
    }

    /// OpenAI-shaped messages including the system turn.
    pub fn openai_messages(&self) -> Vec<Value> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if let Some(system) = &self.system
            && !system.is_empty()
        {
            messages.push(serde_json::json!({"role": "system", "content": system}));
        }
        for message in &self.messages {
            messages.push(serde_json::json!({
                "role": message.role,
                "content": message.content,
            }));
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
        prompt.push_str(&capitalize_role(&message.role));
        prompt.push_str(":\n");
        prompt.push_str(message.content.trim());
        prompt.push_str("\n\n");
    }
    prompt.trim_end().to_string()
}

fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => "User".to_string(),
    }
}

/// A complete non-streaming result.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Assistant text.
    pub content: String,
    /// Normalized finish reason (`stop`, `length`, ...).
    pub finish_reason: String,
    /// Token usage, measured or estimated.
    pub usage: Usage,
    /// Raw upstream JSON when available.
    pub raw: Option<Value>,
}

/// One incremental stream item.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Text appended to the assistant message.
    pub delta: String,
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
            finish_reason: None,
            usage: None,
        }
    }

    /// A terminal chunk.
    pub fn finish(reason: impl Into<String>, usage: Option<Usage>) -> Self {
        Self {
            delta: String::new(),
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

/// Non-Unix stub with the same API.
#[cfg(not(unix))]
#[derive(Debug, Clone)]
pub struct ProcessGroupGuard {
    backend: String,
}

#[cfg(not(unix))]
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
            "process-group supervision is only implemented on Unix"
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

/// Extract a stream delta, finish reason, and usage from one upstream event.
pub fn extract_delta(value: &Value) -> (Option<String>, Option<String>, Option<Usage>) {
    let usage = extract_usage(value);
    if let Some(choices) = value.get("choices").and_then(Value::as_array)
        && let Some(choice) = choices.first()
    {
        let delta = choice
            .get("delta")
            .and_then(|delta| {
                delta
                    .get("content")
                    .and_then(content_to_text)
                    .or_else(|| delta.get("reasoning_content").and_then(content_to_text))
            })
            .or_else(|| {
                choice
                    .get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            });
        let finish = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        if delta.is_some() || finish.is_some() {
            return (delta, finish, usage);
        }
    }

    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match event_type {
        "content_block_delta" => {
            let delta = value
                .get("delta")
                .and_then(|delta| delta.get("text"))
                .and_then(Value::as_str)
                .map(ToString::to_string);
            return (delta, None, usage);
        }
        "message_delta" => {
            let finish = value
                .get("delta")
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str)
                .map(ToString::to_string);
            return (None, finish, usage);
        }
        "message_stop" | "done" => return (None, Some("stop".into()), usage),
        "error" => return (None, None, usage),
        _ => {}
    }

    if let Some(delta) = value.get("delta") {
        if let Some(text) = delta.as_str() {
            return (Some(text.to_string()), None, usage);
        }
        if let Some(text) = delta
            .get("text")
            .and_then(Value::as_str)
            .or_else(|| delta.get("content").and_then(Value::as_str))
        {
            return (Some(text.to_string()), None, usage);
        }
    }
    for key in ["content", "text", "token", "response"] {
        if let Some(text) = value.get(key).and_then(Value::as_str) {
            return (Some(text.to_string()), None, usage);
        }
    }
    (None, None, usage)
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

/// Map an upstream HTTP status to a gateway error, redacting the body.
pub fn map_upstream_status(status: u16, body: &str) -> GatewayError {
    let body = telemetry::redact_text(body);
    let body = truncate(&body, 800);
    match status {
        429 => GatewayError::UpstreamRateLimited(format!("upstream returned 429: {body}")),
        400 | 422 => GatewayError::BadRequest(format!("upstream rejected request: {body}")),
        401 | 403 => GatewayError::Upstream(format!("upstream authentication failed: {body}")),
        404 => GatewayError::NotFound(format!("upstream resource not found: {body}")),
        _ => GatewayError::Upstream(format!("upstream returned HTTP {status}: {body}")),
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

/// Collect a stream into a bounded buffer.
pub async fn collect_stream(
    mut stream: ChunkStream,
    max_bytes: usize,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let mut content = String::new();
    let mut finish = None;
    let mut usage = None;
    while let Some(item) = stream.next().await {
        let chunk = item?;
        content.push_str(&chunk.delta);
        if content.len() > max_bytes {
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
    Ok((content, finish, usage))
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
}
