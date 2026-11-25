//! HTTP ingress adapters.
//!
//! The gateway exposes the OpenAI and Anthropic wire formats over the same
//! routing core. Each adapter is responsible only for translating its
//! protocol into an [`ExecutionRequest`] and translating the driver response
//! back. Authentication, model resolution, session affinity, and fallback are
//! shared.

pub mod anthropic;
pub mod openai;
pub mod ops;

use std::{
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
    routing::{get, post},
};
use futures::Stream;
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::{
    auth::Principal,
    driver::{
        ChatMessage, ChunkStream, Completion, DriverResponse, ExecutionRequest, StreamChunk,
        WorkspaceMode, collect_stream, render_prompt,
    },
    error::{GatewayError, Result},
    metrics,
    router::{RoutedResponse, derive_session_key},
    state::AppState,
};

/// Per-request context resolved before dispatch.
#[derive(Debug, Clone)]
pub(crate) struct RequestContext {
    /// Authenticated caller.
    pub principal: Principal,
    /// Public model after default resolution.
    pub model: String,
    /// Correlation id returned to the client.
    pub request_id: String,
    /// Affinity key.
    pub session_key: String,
    /// Requested workspace isolation.
    pub workspace_mode: WorkspaceMode,
}

/// Build the complete axum application.
pub fn build_router(state: Arc<AppState>) -> AxumRouter {
    let config = state.config();
    let body_limit = config.server.max_body_bytes.max(1);
    let cors_permissive = config.server.cors_permissive;

    let mut router = AxumRouter::new()
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/messages", post(anthropic::messages))
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

/// Authenticate the caller, resolve the model, and derive affinity.
pub(crate) fn prepare(
    state: &AppState,
    headers: &HeaderMap,
    requested_model: Option<&str>,
    system: Option<&str>,
    first_user: &str,
) -> Result<RequestContext> {
    let principal = state.tokens().authenticate(headers)?;
    let model = state.resolve_model(requested_model)?;
    if !principal.allows_model(&model) {
        return Err(GatewayError::Forbidden(format!(
            "token {:?} is not allowed to use model {model:?}",
            principal.name
        )));
    }

    let request_id = header_text(headers, "x-request-id")
        .filter(|value| value.len() <= 128 && is_safe_id(value))
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("vouch-{}", Uuid::new_v4().simple()));
    let explicit_session =
        header_text(headers, "x-session-id").or_else(|| header_text(headers, "x-vouch-session-id"));
    let session_key = derive_session_key(explicit_session, system, first_user);
    let workspace_mode = match header_text(headers, "x-workspace-mode") {
        Some(value) => WorkspaceMode::parse(value).ok_or_else(|| {
            GatewayError::BadRequest(format!(
                "invalid x-workspace-mode {value:?}; expected auto, worktree, or none"
            ))
        })?,
        None => WorkspaceMode::Auto,
    };

    tracing::debug!(
        principal = %principal.name,
        model = %model,
        request_id = %request_id,
        "request authenticated"
    );
    Ok(RequestContext {
        principal,
        model,
        request_id,
        session_key,
        workspace_mode,
    })
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
        extra,
    }
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
            let (content, finish_reason, usage) = collect_stream(stream, 16 * 1024 * 1024).await?;
            Ok(Completion {
                content,
                finish_reason: finish_reason.unwrap_or_else(|| "stop".into()),
                usage: usage.unwrap_or_default(),
                raw: None,
            })
        }
    }
}

/// Coerce a driver response into a chunk stream.
pub(crate) fn into_chunk_stream(response: DriverResponse) -> ChunkStream {
    match response {
        DriverResponse::Stream(stream) => stream,
        DriverResponse::Complete(completion) => Box::pin(futures::stream::iter(vec![
            Ok(StreamChunk::delta(completion.content)),
            Ok(StreamChunk::finish(
                completion.finish_reason,
                Some(completion.usage),
            )),
        ])),
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
pub(crate) fn first_user_text(messages: &[ChatMessage]) -> &str {
    messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .unwrap_or("")
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
