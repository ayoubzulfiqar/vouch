//! OpenAI-compatible ingress.
//!
//! Supports `GET /v1/models` and `POST /v1/chat/completions` with either a
//! buffered JSON response or a `text/event-stream` of `chat.completion.chunk`
//! objects terminated by `data: [DONE]`.

use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::Event},
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    driver::{ChatMessage, StreamChunk, Usage, content_to_text, truncate},
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, completion, execute, execution_request, first_user_text, into_chunk_stream,
        prepare, sse_response, unix_now,
    },
    state::AppState,
};

/// OpenAI `POST /v1/chat/completions` request body.
#[derive(Debug, Deserialize)]
pub struct OpenAiChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<Value>,
}

/// List configured public models.
pub async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(error) = state.tokens().authenticate(&headers) {
        return error.into_openai_response();
    }
    let data: Vec<Value> = state
        .router()
        .models()
        .into_iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "vouch",
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

/// OpenAI chat completions, streaming or buffered.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<OpenAiChatRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(rejection) => {
            return GatewayError::BadRequest(format!(
                "invalid OpenAI request body: {}",
                truncate(&rejection.body_text(), 500)
            ))
            .into_openai_response();
        }
    };
    match chat_completions_inner(state, headers, payload).await {
        Ok(response) => response,
        Err(error) => error.into_openai_response(),
    }
}

async fn chat_completions_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: OpenAiChatRequest,
) -> Result<Response> {
    let (system, messages) = normalize_messages(payload.messages);
    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "messages must contain at least one non-system turn".into(),
        ));
    }
    let first_user = first_user_text(&messages).to_string();
    let context = prepare(
        &state,
        &headers,
        payload.model.as_deref(),
        system.as_deref(),
        &first_user,
    )?;
    let max_tokens = payload.max_tokens.or(payload.max_completion_tokens);
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        payload.stream,
        payload.temperature,
        max_tokens,
        payload.extra,
        cancel.clone(),
    );
    let routed = execute(&state, request).await?;
    let backend = routed.backend_id.clone();
    let fallback = routed.fallback_used;
    let driver_response = routed.response;

    if payload.stream {
        let config = state.config();
        let keep_alive = config.server.keep_alive_interval_secs;
        let stream_timeout = config.server.stream_timeout_secs.max(1);
        let mut chunks = into_chunk_stream(driver_response);
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let model = context.model.clone();
        let request_id = context.request_id.clone();
        let stream_cancel = cancel.clone();

        let stream = async_stream::stream! {
            let _guard = CancelOnDrop(stream_cancel.clone());
            let created = unix_now();
            let role = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null,
                }],
            });
            yield Ok::<Event, Infallible>(Event::default().data(role.to_string()));

            let deadline = tokio::time::Instant::now() + Duration::from_secs(stream_timeout);
            let mut timed_out = false;
            loop {
                let item = tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        None
                    }
                    item = chunks.next() => item,
                };
                if timed_out {
                    stream_cancel.cancel();
                    let error = json!({
                        "error": {
                            "message": format!("stream timed out after {stream_timeout} seconds"),
                            "type": "api_error",
                            "code": "stream_timeout",
                        }
                    });
                    yield Ok(Event::default().data(error.to_string()));
                    break;
                }
                match item {
                    Some(Ok(chunk)) => {
                        let StreamChunk { delta, finish_reason, usage } = chunk;
                        if !delta.is_empty() {
                            let payload = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": null,
                                }],
                            });
                            yield Ok(Event::default().data(payload.to_string()));
                        }
                        if let Some(reason) = finish_reason {
                            let mut payload = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": reason,
                                }],
                            });
                            if let Some(usage) = usage
                                && let Some(object) = payload.as_object_mut()
                            {
                                object.insert("usage".into(), usage_json(&usage));
                            }
                            yield Ok(Event::default().data(payload.to_string()));
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        let payload = json!({
                            "error": {
                                "message": error.to_string(),
                                "type": "api_error",
                                "code": "stream_error",
                            }
                        });
                        yield Ok(Event::default().data(payload.to_string()));
                        break;
                    }
                    None => {
                        let payload = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        });
                        yield Ok(Event::default().data(payload.to_string()));
                        break;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };

        let mut response = sse_response(stream, keep_alive);
        add_routing_headers_from_parts(&mut response, &backend, fallback, &request_id);
        return Ok(response);
    }

    let completion = completion(driver_response).await?;
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_now(),
        "model": context.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion.content},
            "finish_reason": completion.finish_reason,
        }],
        "usage": usage_json(&completion.usage),
    });
    let mut response = (StatusCode::OK, Json(body)).into_response();
    add_routing_headers_from_parts(&mut response, &backend, fallback, &context.request_id);
    Ok(response)
}

fn normalize_messages(raw: Vec<OpenAiMessage>) -> (Option<String>, Vec<ChatMessage>) {
    let mut system_parts = Vec::new();
    let mut messages = Vec::with_capacity(raw.len());
    for message in raw {
        let role = message.role.trim();
        let role = if role.is_empty() { "user" } else { role };
        let content = message
            .content
            .as_ref()
            .and_then(content_to_text)
            .unwrap_or_default();
        if role == "system" || role == "developer" {
            if !content.trim().is_empty() {
                system_parts.push(content);
            }
        } else {
            messages.push(ChatMessage {
                role: role.to_string(),
                content,
            });
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, messages)
}

fn usage_json(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    })
}

fn add_routing_headers_from_parts(
    response: &mut Response,
    backend: &str,
    fallback: bool,
    request_id: &str,
) {
    crate::ingress::insert_header(response, "x-vouch-backend", backend);
    crate::ingress::insert_header(
        response,
        "x-vouch-fallback",
        if fallback { "true" } else { "false" },
    );
    crate::ingress::insert_header(response, "x-vouch-request-id", request_id);
}
