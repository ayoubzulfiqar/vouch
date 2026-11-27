//! HTTP ingress adapters.
//!
//! The gateway exposes the OpenAI and Anthropic wire formats over the same
//! routing core. Each adapter is responsible only for translating its
//! protocol into an [`ExecutionRequest`] and translating the driver response
//! back. Authentication, model resolution, session affinity, and fallback are
//! shared.

pub mod admin;
pub mod anthropic;
pub mod gemini;
pub mod ollama;
pub mod openai;
pub mod ops;
pub mod responses;
pub mod ui;

use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router as AxumRouter,
    body::Body,
    extract::{DefaultBodyLimit, MatchedPath},
    http::{HeaderMap, HeaderName, HeaderValue, Request},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use futures::Stream;
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::{
    auth::{Principal, tokens::TokenConcurrencyGuard},
    cache::{CACHE_STATUS_HEADER, CacheLookup, synthesize_stream_from_completion},
    driver::{
        ChatMessage, ChunkStream, Completion, ContentPart, DriverResponse, ExecutionRequest,
        MessageContent, ResponseFormat, StreamChunk, ThinkingBlock, ToolCall, ToolCallDelta, Usage,
        WorkspaceMode, attachments::parse_data_uri, collect_stream_full, render_prompt,
    },
    error::{GatewayError, Result},
    metrics,
    router::{RoutedResponse, RoutingInput, derive_session_key},
    state::AppState,
    telemetry::{ActiveSpan, SpanKind, TraceContext},
    usage::count_tokens,
};

/// Per-request context resolved before dispatch.
#[derive(Debug, Clone)]
pub(crate) struct RequestContext {
    /// Authenticated caller.
    pub principal: Principal,
    /// Resolved canonical route name.
    pub model: String,
    /// Requested model/alias name before rule/alias expansion.
    pub requested_model: String,
    /// Correlation id returned to the client.
    pub request_id: String,
    /// Affinity key.
    pub session_key: String,
    /// Requested workspace isolation.
    pub workspace_mode: WorkspaceMode,
    /// Propagated W3C trace context for spans and upstream headers.
    pub trace_context: TraceContext,
    /// Estimated prompt BPE tokens computed at ingress.
    pub prompt_tokens: u64,
    /// Per-token concurrency permit released when response/stream drops.
    #[allow(dead_code)]
    pub concurrency_guard: Arc<TokenConcurrencyGuard>,
}

/// Dispatched route result paired with optional `X-Vouch-Cache` header and miss key.
pub(crate) struct DispatchedExecution {
    pub routed: RoutedResponse,
    pub cache_status: Option<HeaderValue>,
    pub cache_miss_key: Option<String>,
}

/// RAII accounting guard for streaming responses (SSE, WS, NDJSON, Gemini JSON array).
///
/// Counts emitted text and tool-call bytes via `tiktoken` if a stream closes early
/// (e.g., client disconnect `408` or stream timeout `504`) and debits [`UsageTracker`]
/// plus daily/TPM token budgets exactly once on completion or drop.
pub(crate) struct StreamUsageGuard {
    state: Arc<AppState>,
    context: RequestContext,
    backend_id: String,
    fallback: bool,
    started: Instant,
    ingress_span: Option<ActiveSpan>,
    cache_miss_key: Option<String>,
    emitted_text: String,
    emitted_thinking: Vec<ThinkingBlock>,
    emitted_tool_bytes: String,
    emitted_tool_calls: BTreeMap<usize, ToolCall>,
    explicit_usage: Option<Usage>,
    committed: bool,
}

impl StreamUsageGuard {
    /// Create an armed accounting guard wrapping an active stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        state: Arc<AppState>,
        context: RequestContext,
        backend_id: String,
        fallback: bool,
        started: Instant,
        ingress_span: Option<ActiveSpan>,
        cache_miss_key: Option<String>,
    ) -> Self {
        Self {
            state,
            context,
            backend_id,
            fallback,
            started,
            ingress_span,
            cache_miss_key,
            emitted_text: String::new(),
            emitted_thinking: Vec::new(),
            emitted_tool_bytes: String::new(),
            emitted_tool_calls: BTreeMap::new(),
            explicit_usage: None,
            committed: false,
        }
    }

    /// Observe an upstream [`StreamChunk`] before rendering it to the wire format.
    pub(crate) fn observe_chunk(&mut self, chunk: &StreamChunk) {
        if !chunk.delta.is_empty() {
            self.emitted_text.push_str(&chunk.delta);
        }
        if let Some(thinking) = &chunk.thinking_delta
            && !thinking.is_empty()
        {
            self.emitted_tool_bytes.push_str(thinking);
            match self.emitted_thinking.last_mut() {
                Some(ThinkingBlock::Thinking { thinking: prev, .. }) => prev.push_str(thinking),
                _ => self.emitted_thinking.push(ThinkingBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: None,
                }),
            }
        }
        if let Some(sig) = &chunk.signature_delta
            && let Some(ThinkingBlock::Thinking { signature, .. }) =
                self.emitted_thinking.last_mut()
        {
            signature.get_or_insert_with(String::new).push_str(sig);
        }
        if let Some(redacted) = &chunk.redacted_thinking {
            self.emitted_thinking.push(ThinkingBlock::RedactedThinking {
                data: redacted.clone(),
            });
        }
        for delta in &chunk.tool_calls {
            let entry = self
                .emitted_tool_calls
                .entry(delta.index)
                .or_insert_with(|| ToolCall {
                    id: delta
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple())),
                    name: delta.name.clone().unwrap_or_default(),
                    arguments: String::new(),
                });
            if let Some(id) = &delta.id {
                entry.id = id.clone();
            }
            if let Some(name) = &delta.name {
                entry.name = name.clone();
                self.emitted_tool_bytes.push_str(name);
            }
            if let Some(args) = &delta.arguments_delta {
                entry.arguments.push_str(args);
                self.emitted_tool_bytes.push_str(args);
            }
        }
        if let Some(usage) = chunk.usage {
            self.explicit_usage = Some(usage);
        }
    }

    fn estimate_emitted_usage(&self) -> Usage {
        let mut combined =
            String::with_capacity(self.emitted_text.len() + self.emitted_tool_bytes.len());
        combined.push_str(&self.emitted_text);
        if !self.emitted_tool_bytes.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(&self.emitted_tool_bytes);
        }
        let completion_tokens = count_tokens(&self.context.model, &combined);
        Usage::new(self.context.prompt_tokens, completion_tokens)
    }

    /// Commit normal stream completion once and populate the deterministic cache on a miss.
    pub(crate) fn complete(&mut self, usage: Option<Usage>, finish_reason: &str) -> Usage {
        let mut effective = usage
            .or(self.explicit_usage)
            .unwrap_or_else(|| self.estimate_emitted_usage());
        if effective.prompt_tokens == 0 && self.context.prompt_tokens > 0 {
            effective = Usage::new(self.context.prompt_tokens, effective.completion_tokens);
        }
        if effective.completion_tokens == 0
            && (!self.emitted_text.is_empty() || !self.emitted_tool_bytes.is_empty())
        {
            let est = self.estimate_emitted_usage();
            effective = Usage::new(effective.prompt_tokens, est.completion_tokens);
        }
        if self.committed {
            return effective;
        }
        self.committed = true;

        record_completion_outcome(
            &self.state,
            &self.context,
            &self.backend_id,
            self.fallback,
            effective,
            self.started.elapsed(),
            self.ingress_span.take(),
        );
        if let Some(key) = self.cache_miss_key.take() {
            let cached = Completion {
                content: self.emitted_text.clone(),
                finish_reason: finish_reason.to_string(),
                usage: effective,
                raw: None,
                tool_calls: self.emitted_tool_calls.values().cloned().collect(),
                thinking: self.emitted_thinking.clone(),
            };
            self.state.cache().store(key, &self.backend_id, &cached);
        }
        effective
    }

    /// Record stream timeout / error once while still billing emitted partial tokens.
    pub(crate) fn fail(&mut self, error: &str) {
        if self.committed {
            return;
        }
        self.committed = true;
        let usage = self.estimate_emitted_usage();
        let has_emitted = !self.emitted_text.is_empty() || !self.emitted_tool_bytes.is_empty();
        if has_emitted && usage.total_tokens > 0 {
            self.context.principal.record_tokens(usage.total_tokens);
            let cost = self
                .state
                .calculate_cost_usd(&self.context.model, &self.backend_id, usage);
            self.state.usage().record_outcome_with_cost(
                &self.context.principal.name,
                &self.backend_id,
                usage,
                self.started.elapsed(),
                cost,
                true,
            );
        } else {
            self.state.usage().record_error(
                &self.context.principal.name,
                Some(&self.backend_id),
                self.started.elapsed(),
            );
        }
        if let Some(span) = self.ingress_span.take() {
            span.finish_err(error);
        }
    }
}

impl Drop for StreamUsageGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.committed = true;
            let has_emitted = !self.emitted_text.is_empty() || !self.emitted_tool_bytes.is_empty();
            let usage = if has_emitted {
                self.estimate_emitted_usage()
            } else {
                Usage::new(self.context.prompt_tokens, 0)
            };
            self.context.principal.record_tokens(usage.total_tokens);
            let cost = self
                .state
                .calculate_cost_usd(&self.context.model, &self.backend_id, usage);
            self.state.usage().record_outcome_with_cost(
                &self.context.principal.name,
                &self.backend_id,
                usage,
                self.started.elapsed(),
                cost,
                true,
            );
            if let Some(span) = self.ingress_span.take() {
                span.finish_err("stream dropped before completion");
            }
        }
    }
}

/// Build the complete axum application.
pub fn build_router(state: Arc<AppState>) -> AxumRouter {
    let config = state.config();
    let body_limit = config.server.max_body_bytes.max(1);
    let cors_permissive = config.server.cors_permissive;

    let mut router = AxumRouter::new()
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route(
            "/v1/responses",
            post(responses::responses).get(responses::responses_ws),
        )
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1beta/models/{*model_action}", post(gemini::generate))
        .route("/api/tags", get(ollama::tags))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/admin/reload", post(admin::reload))
        .route(
            "/admin/circuits/{backend}/reset",
            post(admin::reset_circuit),
        )
        .route("/admin/cache", delete(admin::purge_cache))
        .route("/v1/usage", get(ops::usage))
        .route("/mcp", post(crate::cli::mcp::http_mcp))
        .route("/ui", get(ui::dashboard))
        .route("/healthz", get(ops::healthz))
        .route("/readyz", get(ops::readyz))
        .route("/metrics", get(ops::metrics))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(middleware::from_fn(metrics_middleware))
        .with_state(state);

    if cors_permissive {
        router = router.layer(CorsLayer::permissive());
    }
    router
}

/// Authenticate caller, evaluate `[[routing_rules]]` & `[aliases]`, check
/// token allowlist against both requested alias & resolved route, and derive
/// affinity + W3C trace context.
pub(crate) fn prepare_routed(
    state: &AppState,
    headers: &HeaderMap,
    requested_model: Option<&str>,
    system: Option<&str>,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Result<RequestContext> {
    prepare_routed_with_key_hint(
        state,
        headers,
        None,
        requested_model,
        system,
        messages,
        tools,
    )
}

/// Variant of [`prepare_routed`] accepting an optional query-string API key (Gemini `?key=`).
pub(crate) fn prepare_routed_with_key_hint(
    state: &AppState,
    headers: &HeaderMap,
    key_hint: Option<&str>,
    requested_model: Option<&str>,
    system: Option<&str>,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Result<RequestContext> {
    let first_user = first_user_text(messages);
    let has_images = messages_have_images(messages);
    let model_hint = requested_model
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or("");
    let rendered = render_prompt(system, messages);
    let prompt_tokens = count_tokens(model_hint, &rendered);

    let resolved = state.resolve_route(&RoutingInput {
        requested_model,
        has_tools: !tools.is_empty(),
        has_images,
        prompt_tokens,
        headers,
    })?;
    let principal = state.tokens().authorize_route_with_key_hint(
        headers,
        key_hint,
        &resolved.requested,
        &resolved.route,
    )?;
    let concurrency_guard = principal.acquire_concurrency_permit()?;

    let request_id = header_text(headers, "x-request-id")
        .filter(|value| value.len() <= 128 && is_safe_id(value))
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("vouch-{}", Uuid::new_v4().simple()));
    let explicit_session =
        header_text(headers, "x-session-id").or_else(|| header_text(headers, "x-vouch-session-id"));
    let session_key = derive_session_key(explicit_session, system, &first_user);
    let workspace_mode = match header_text(headers, "x-workspace-mode") {
        Some(value) => WorkspaceMode::parse(value).ok_or_else(|| {
            GatewayError::BadRequest(format!(
                "invalid x-workspace-mode {value:?}; expected auto, worktree, or none"
            ))
        })?,
        None => WorkspaceMode::Auto,
    };
    let trace_context = TraceContext::from_headers(headers);

    tracing::debug!(
        principal = %principal.name,
        requested_model = %resolved.requested,
        route = %resolved.route,
        matched_rule = ?resolved.matched_rule,
        request_id = %request_id,
        "request authenticated"
    );
    Ok(RequestContext {
        principal,
        model: resolved.route,
        requested_model: resolved.requested,
        request_id,
        session_key,
        workspace_mode,
        trace_context,
        prompt_tokens,
        concurrency_guard,
    })
}

fn messages_have_images(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|msg| match &msg.content {
        MessageContent::Text(_) => false,
        MessageContent::Parts(parts) => parts.iter().any(|part| {
            matches!(
                part,
                ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. }
            )
        }),
    })
}

/// Start an OTLP `ingress` span when `[telemetry].otlp_endpoint` is active.
pub(crate) fn start_ingress_span(
    state: &AppState,
    context: &RequestContext,
    endpoint: &'static str,
) -> ActiveSpan {
    let mut span =
        state
            .otel()
            .start_span(context.trace_context.clone(), "ingress", SpanKind::Server);
    span.set_str("vouch.route", &context.model);
    span.set_str("vouch.requested_model", &context.requested_model);
    span.set_str("http.route", endpoint);
    span
}

/// Record token daily budget + usage analytics + OTLP span completion on success.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_completion_outcome(
    state: &AppState,
    context: &RequestContext,
    backend: &str,
    fallback: bool,
    usage: Usage,
    elapsed: Duration,
    span: Option<ActiveSpan>,
) {
    let effective = if usage.total_tokens == 0 && context.prompt_tokens > 0 {
        Usage::new(context.prompt_tokens, usage.completion_tokens)
    } else {
        usage
    };
    context.principal.record_tokens(effective.total_tokens);
    let cost_usd = state.calculate_cost_usd(&context.model, backend, effective);
    state.usage().record_success_with_cost(
        &context.principal.name,
        backend,
        effective,
        elapsed,
        cost_usd,
    );
    if let Some(mut span) = span {
        span.set_str("vouch.route", &context.model);
        span.set_str("vouch.backend", backend);
        span.set_bool("vouch.fallback", fallback);
        span.set_int("gen_ai.usage.input_tokens", effective.prompt_tokens as i64);
        span.set_int(
            "gen_ai.usage.output_tokens",
            effective.completion_tokens as i64,
        );
        span.finish_ok();
    }
}

/// Record failure analytics + OTLP span error cleanly.
pub(crate) fn record_error_outcome(
    state: &AppState,
    context: &RequestContext,
    backend: Option<&str>,
    elapsed: Duration,
    error: &str,
    span: Option<ActiveSpan>,
) {
    state
        .usage()
        .record_error(&context.principal.name, backend, elapsed);
    if let Some(span) = span {
        span.finish_err(error);
    }
}

/// Build a normalized execution request from an adapter payload.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execution_request(
    context: &RequestContext,
    system: Option<String>,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    tools: Vec<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    response_format: Option<ResponseFormat>,
    extra: serde_json::Map<String, serde_json::Value>,
    cancel: CancellationToken,
) -> ExecutionRequest {
    let prompt = render_prompt(system.as_deref(), &messages);
    tracing::trace!(token = %context.principal.name, "building execution request");
    ExecutionRequest {
        request_id: context.request_id.clone(),
        session_key: context.session_key.clone(),
        model: context.model.clone(),
        system,
        messages,
        prompt,
        stream,
        temperature,
        max_tokens,
        workspace_mode: context.workspace_mode,
        workdir: None,
        cancel,
        tools,
        tool_choice,
        response_format,
        backend_session_id: None,
        delta_messages: None,
        session_capture: Arc::new(parking_lot::Mutex::new(None)),
        trace_context: Some(context.trace_context.clone()),
        profile_id: None,
        profile_env: BTreeMap::new(),
        allowed_backends: context.principal.allowed_backends(),
        extra,
    }
}

/// Dispatch a request with deterministic response cache lookup and router deadline.
pub(crate) async fn execute_or_cached(
    state: &AppState,
    context: &RequestContext,
    headers: &HeaderMap,
    request: ExecutionRequest,
) -> Result<DispatchedExecution> {
    let workspace_isolated = match context.workspace_mode {
        WorkspaceMode::Worktree => true,
        WorkspaceMode::None => false,
        WorkspaceMode::Auto => state.workspaces().enabled(),
    };
    let lookup = state
        .cache()
        .lookup(&context.model, &request, headers, workspace_isolated);
    let cache_status = lookup.header_value();
    let cache_miss_key = lookup.miss_key().map(ToString::to_string);

    match lookup {
        CacheLookup::Hit(cached) => {
            let response = if request.stream {
                DriverResponse::Stream(synthesize_stream_from_completion(&cached.completion))
            } else {
                DriverResponse::Complete(cached.completion)
            };
            Ok(DispatchedExecution {
                routed: RoutedResponse {
                    backend_id: cached.backend_id,
                    fallback_used: false,
                    response,
                    attempts: Vec::new(),
                },
                cache_status,
                cache_miss_key: None,
            })
        }
        CacheLookup::Miss(_) | CacheLookup::Bypass | CacheLookup::Disabled => {
            let routed = execute(state, request).await?;
            Ok(DispatchedExecution {
                routed,
                cache_status,
                cache_miss_key,
            })
        }
    }
}

/// Attach `x-vouch-cache` when `[cache].enabled` returned a cache status header.
pub(crate) fn insert_cache_header(response: &mut Response, status: Option<HeaderValue>) {
    if let Some(value) = status {
        response.headers_mut().insert(CACHE_STATUS_HEADER, value);
    }
}

/// Attach standard `x-vouch-*` routing and cache headers onto any egress HTTP response.
pub(crate) fn insert_routing_headers(
    response: &mut Response,
    backend: &str,
    fallback: bool,
    request_id: &str,
    cache_status: Option<HeaderValue>,
) {
    insert_header(response, "x-vouch-backend", backend);
    insert_header(
        response,
        "x-vouch-fallback",
        if fallback { "true" } else { "false" },
    );
    insert_header(response, "x-vouch-request-id", request_id);
    insert_cache_header(response, cache_status);
}

/// Parse an RFC-3339-like UTC `YYYY-MM-DDTHH:MM:SSZ` timestamp for current time.
pub(crate) fn iso8601_utc_now() -> String {
    let secs = unix_now();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    // Civil calendar conversion from Unix epoch days (Howard Hinnant algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(m <= 2);
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Dispatch a request with the configured non-streaming deadline.
pub(crate) async fn execute(state: &AppState, request: ExecutionRequest) -> Result<RoutedResponse> {
    let timeout = Duration::from_secs(state.config().server.request_timeout_secs.max(1));
    let workspace_mode = request.workspace_mode;
    tokio::time::timeout(
        timeout,
        state
            .router()
            .execute(request, workspace_mode, state.workspaces().as_ref()),
    )
    .await
    .map_err(|_| GatewayError::Timeout(timeout))?
}

/// Coerce a driver response into a complete result.
pub(crate) async fn completion(response: DriverResponse) -> Result<Completion> {
    match response {
        DriverResponse::Complete(completion) => Ok(completion),
        DriverResponse::Stream(stream) => {
            let collected = collect_stream_full(stream, 16 * 1024 * 1024).await?;
            let default_reason = if collected.tool_calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            };
            Ok(Completion {
                content: collected.content,
                finish_reason: collected
                    .finish_reason
                    .unwrap_or_else(|| default_reason.into()),
                usage: collected.usage.unwrap_or_default(),
                raw: None,
                tool_calls: collected.tool_calls,
                thinking: collected.thinking,
            })
        }
    }
}

/// Coerce a driver response into a chunk stream.
pub(crate) fn into_chunk_stream(response: DriverResponse) -> ChunkStream {
    match response {
        DriverResponse::Stream(stream) => stream,
        DriverResponse::Complete(completion) => {
            let mut items = Vec::new();
            for block in completion.thinking {
                match block {
                    ThinkingBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        let mut chunk = StreamChunk::thinking(thinking);
                        chunk.signature_delta = signature;
                        items.push(Ok(chunk));
                    }
                    ThinkingBlock::RedactedThinking { data } => {
                        let mut chunk = StreamChunk::delta("");
                        chunk.redacted_thinking = Some(data);
                        items.push(Ok(chunk));
                    }
                }
            }
            if !completion.content.is_empty() {
                items.push(Ok(StreamChunk::delta(completion.content)));
            }
            if !completion.tool_calls.is_empty() {
                let deltas = completion
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, call)| ToolCallDelta {
                        index,
                        id: Some(call.id),
                        name: Some(call.name),
                        arguments_delta: Some(call.arguments),
                    })
                    .collect();
                items.push(Ok(StreamChunk::tool_calls(deltas)));
            }
            items.push(Ok(StreamChunk::finish(
                completion.finish_reason,
                Some(completion.usage),
            )));
            Box::pin(futures::stream::iter(items))
        }
    }
}

/// Insert a header when the value is a valid header value.
pub(crate) fn insert_header(response: &mut Response, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(name), value);
    }
}

/// Read a header as trimmed UTF-8.
pub(crate) fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Whether a client-supplied correlation id is safe to echo.
fn is_safe_id(value: &str) -> bool {
    value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
}

/// First user turn, used for prompt-derived affinity.
pub(crate) fn first_user_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_plain_text())
        .unwrap_or_default()
}

/// Parse OpenAI, Anthropic, or Responses JSON `"content"` / item values into
/// [`MessageContent`], preserving base64/URL images and `cache_control` hints.
pub(crate) fn parse_json_content(value: &serde_json::Value) -> MessageContent {
    match value {
        serde_json::Value::String(text) => MessageContent::Text(text.clone()),
        serde_json::Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                append_parsed_part(item, &mut parts);
            }
            collapse_parts(parts)
        }
        serde_json::Value::Object(_) => {
            let mut parts = Vec::new();
            append_parsed_part(value, &mut parts);
            collapse_parts(parts)
        }
        serde_json::Value::Null => MessageContent::Text(String::new()),
        other => MessageContent::Text(other.to_string()),
    }
}

fn collapse_parts(parts: Vec<ContentPart>) -> MessageContent {
    let any_rich = parts.iter().any(|part| match part {
        ContentPart::Text { cache_control, .. } => cache_control.is_some(),
        ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => true,
    });
    if !any_rich {
        let joined: String = parts
            .into_iter()
            .filter_map(|p| match p {
                ContentPart::Text { text, .. } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return MessageContent::Text(joined);
    }
    MessageContent::Parts(parts)
}

fn append_parsed_part(item: &serde_json::Value, parts: &mut Vec<ContentPart>) {
    if let Some(text) = item.as_str() {
        parts.push(ContentPart::Text {
            text: text.to_string(),
            cache_control: None,
        });
        return;
    }
    let item_type = item.get("type").and_then(serde_json::Value::as_str);
    match item_type {
        Some("image_url" | "input_image") => {
            let url = item
                .get("image_url")
                .and_then(|v| v.as_str().or_else(|| v.get("url").and_then(|u| u.as_str())))
                .or_else(|| item.get("url").and_then(|u| u.as_str()));
            if let Some(url) = url {
                if let Some((media_type, data)) = parse_data_uri(url) {
                    parts.push(ContentPart::ImageBase64 { media_type, data });
                } else {
                    parts.push(ContentPart::ImageUrl {
                        url: url.to_string(),
                    });
                }
            }
        }
        Some("image") => {
            if let Some(source) = item.get("source") {
                match source.get("type").and_then(serde_json::Value::as_str) {
                    Some("base64") => {
                        if let Some(data) = source.get("data").and_then(serde_json::Value::as_str) {
                            let media_type = source
                                .get("media_type")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("image/png")
                                .to_string();
                            parts.push(ContentPart::ImageBase64 {
                                media_type,
                                data: data.to_string(),
                            });
                        }
                    }
                    Some("url") => {
                        if let Some(url) = source.get("url").and_then(serde_json::Value::as_str) {
                            if let Some((media_type, data)) = parse_data_uri(url) {
                                parts.push(ContentPart::ImageBase64 { media_type, data });
                            } else {
                                parts.push(ContentPart::ImageUrl {
                                    url: url.to_string(),
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {
            if let Some(text) = item
                .get("text")
                .and_then(serde_json::Value::as_str)
                .or_else(|| item.get("content").and_then(serde_json::Value::as_str))
            {
                parts.push(ContentPart::Text {
                    text: text.to_string(),
                    cache_control: item.get("cache_control").cloned(),
                });
            }
        }
    }
}

/// Current Unix timestamp in seconds.
pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Cancels the request token when the downstream stream is dropped.
pub(crate) struct CancelOnDrop(pub CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Build an SSE response with optional keep-alive comments.
pub(crate) fn sse_response<S>(stream: S, keep_alive_secs: u64) -> Response
where
    S: Stream<Item = std::result::Result<Event, Infallible>> + Send + 'static,
{
    if keep_alive_secs == 0 {
        let mut response = Sse::new(stream).into_response();
        insert_header(&mut response, "cache-control", "no-cache");
        insert_header(&mut response, "x-accel-buffering", "no");
        response
    } else {
        let mut response = Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(keep_alive_secs))
                    .text("vouch"),
            )
            .into_response();
        insert_header(&mut response, "cache-control", "no-cache");
        insert_header(&mut response, "x-accel-buffering", "no");
        response
    }
}

async fn metrics_middleware(request: Request<Body>, next: Next) -> Response {
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());
    let protocol = if path == "/v1/messages" {
        "anthropic"
    } else if path == "/mcp" {
        "mcp"
    } else if path.starts_with("/v1beta/") {
        "gemini"
    } else if path.starts_with("/api/") {
        "ollama"
    } else if path.starts_with("/admin/") {
        "admin"
    } else if path.starts_with("/v1/") {
        "openai"
    } else {
        "ops"
    };
    let started = Instant::now();
    let response = next.run(request).await;
    metrics::record_request(
        protocol,
        &path,
        response.status().as_u16(),
        started.elapsed(),
    );
    response
}
