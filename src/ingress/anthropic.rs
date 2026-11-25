//! Anthropic Messages-compatible ingress.
//!
//! The adapter accepts `system`, `messages`, `max_tokens`, and streaming
//! requests, translates them into the shared normalized execution request, and
//! renders either an Anthropic `message` object or the documented SSE event
//! sequence (`message_start`, `content_block_delta`, `message_delta`,
//! `message_stop`).

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
    driver::{ChatMessage, StreamChunk, content_to_text, truncate},
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, completion, execute, execution_request, first_user_text, into_chunk_stream,
        prepare, sse_response,
    },
    state::AppState,
};

/// Anthropic `POST /v1/messages` request body.
#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<Value>,
}

/// Anthropic Messages endpoint.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<AnthropicRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(rejection) => {
            return GatewayError::BadRequest(format!(
                "invalid Anthropic request body: {}",
                truncate(&rejection.body_text(), 500)
            ))
            .into_anthropic_response();
        }
    };
    match messages_inner(state, headers, payload).await {
        Ok(response) => response,
        Err(error) => error.into_anthropic_response(),
    }
}

async fn messages_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: AnthropicRequest,
) -> Result<Response> {
    let max_tokens = payload.max_tokens.ok_or_else(|| {
        GatewayError::BadRequest("max_tokens is required for the Anthropic Messages API".into())
    })?;
    if max_tokens == 0 {
        return Err(GatewayError::BadRequest(
            "max_tokens must be greater than zero".into(),
        ));
    }

    let (system, messages) = normalize_messages(payload.system, payload.messages);
    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "messages must contain at least one turn".into(),
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
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        payload.stream,
        payload.temperature,
        Some(max_tokens),
        payload.extra,
        cancel.clone(),
    );
    let routed = execute(&state, request).await?;
    let backend = routed.backend_id.clone();
    let fallback = routed.fallback_used;
    let driver_response = routed.response;
    let message_id = format!("msg_{}", Uuid::new_v4().simple());

    if payload.stream {
        let config = state.config();
        let keep_alive = config.server.keep_alive_interval_secs;
        let stream_timeout = config.server.stream_timeout_secs.max(1);
        let mut chunks = into_chunk_stream(driver_response);
        let model = context.model.clone();
        let request_id = context.request_id.clone();
        let stream_cancel = cancel.clone();

        let stream = async_stream::stream! {
            let _guard = CancelOnDrop(stream_cancel.clone());
            let start = json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            });
            yield Ok::<Event, Infallible>(
                Event::default().event("message_start").data(start.to_string())
            );
            let block_start = json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            });
            yield Ok(Event::default().event("content_block_start").data(block_start.to_string()));

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
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("stream timed out after {stream_timeout} seconds"),
                        }
                    });
                    yield Ok(Event::default().event("error").data(error.to_string()));
                    break;
                }
                match item {
                    Some(Ok(chunk)) => {
                        let StreamChunk { delta, finish_reason, usage } = chunk;
                        if !delta.is_empty() {
                            let payload = json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": delta},
                            });
                            yield Ok(Event::default()
                                .event("content_block_delta")
                                .data(payload.to_string()));
                        }
                        if let Some(reason) = finish_reason {
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": 0}).to_string()));
                            let output_tokens = usage.map(|usage| usage.completion_tokens).unwrap_or(0);
                            let delta = json!({
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": anthropic_stop_reason(&reason),
                                    "stop_sequence": null,
                                },
                                "usage": {"output_tokens": output_tokens},
                            });
                            yield Ok(Event::default().event("message_delta").data(delta.to_string()));
                            yield Ok(Event::default()
                                .event("message_stop")
                                .data(json!({"type": "message_stop"}).to_string()));
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        let payload = json!({
                            "type": "error",
                            "error": {"type": "api_error", "message": error.to_string()},
                        });
                        yield Ok(Event::default().event("error").data(payload.to_string()));
                        break;
                    }
                    None => {
                        yield Ok(Event::default()
                            .event("content_block_stop")
                            .data(json!({"type": "content_block_stop", "index": 0}).to_string()));
                        let delta = json!({
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                            "usage": {"output_tokens": 0},
                        });
                        yield Ok(Event::default().event("message_delta").data(delta.to_string()));
                        yield Ok(Event::default()
                            .event("message_stop")
                            .data(json!({"type": "message_stop"}).to_string()));
                        break;
                    }
                }
            }
        };

        let mut response = sse_response(stream, keep_alive);
        crate::ingress::insert_header(&mut response, "x-vouch-backend", &backend);
        crate::ingress::insert_header(
            &mut response,
            "x-vouch-fallback",
            if fallback { "true" } else { "false" },
        );
        crate::ingress::insert_header(&mut response, "x-vouch-request-id", &request_id);
        return Ok(response);
    }

    let completion = completion(driver_response).await?;
    let body = json!({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": context.model,
        "content": [{"type": "text", "text": completion.content}],
        "stop_reason": anthropic_stop_reason(&completion.finish_reason),
        "stop_sequence": null,
        "usage": {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        },
    });
    let mut response = (StatusCode::OK, Json(body)).into_response();
    crate::ingress::insert_header(&mut response, "x-vouch-backend", &backend);
    crate::ingress::insert_header(
        &mut response,
        "x-vouch-fallback",
        if fallback { "true" } else { "false" },
    );
    crate::ingress::insert_header(&mut response, "x-vouch-request-id", &context.request_id);
    Ok(response)
}

fn normalize_messages(
    system: Option<Value>,
    raw: Vec<AnthropicMessage>,
) -> (Option<String>, Vec<ChatMessage>) {
    let mut system_parts = Vec::new();
    if let Some(system) = system
        && let Some(text) = content_to_text(&system)
        && !text.trim().is_empty()
    {
        system_parts.push(text);
    }
    let mut messages = Vec::with_capacity(raw.len());
    for message in raw {
        let role = message.role.trim();
        let role = if role.is_empty() { "user" } else { role };
        let content = message
            .content
            .as_ref()
            .and_then(content_to_text)
            .unwrap_or_default();
        if role == "system" {
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

fn anthropic_stop_reason(reason: &str) -> &'static str {
    match reason {
        "length" | "max_tokens" => "max_tokens",
        "tool_calls" | "tool_use" => "tool_use",
        "stop_sequence" => "stop_sequence",
        _ => "end_turn",
    }
}
