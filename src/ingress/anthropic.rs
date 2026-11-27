//! Anthropic Messages-compatible ingress.
//!
//! The adapter accepts `system`, `messages`, `max_tokens`, `thinking`,
//! `cache_control`, multimodal `image` blocks, and streaming requests,
//! translating them into the shared normalized execution request and rendering
//! either an Anthropic `message` object or the standard SSE event sequence.

use std::{collections::BTreeMap, convert::Infallible, sync::Arc, time::Duration};

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
    driver::{
        ChatMessage, ContentPart, MessageContent, StreamChunk, ThinkingBlock, ToolCall,
        tool_bridge::normalize_tools, truncate,
    },
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, completion, execute, execution_request, first_user_text, into_chunk_stream,
        parse_json_content, prepare, sse_response,
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
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
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

    let mut extra = payload.extra;
    if let Some(thinking) = &payload.thinking {
        validate_thinking_config(thinking)?;
        extra.insert("thinking".into(), thinking.clone());
    }
    if let Some(system_val) = &payload.system
        && system_val.is_array()
    {
        extra.insert("anthropic_system_blocks".into(), system_val.clone());
    }

    let (system, messages) = normalize_messages(payload.system, payload.messages);
    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "messages must contain at least one turn".into(),
        ));
    }
    let first_user = first_user_text(&messages);
    let context = prepare(
        &state,
        &headers,
        payload.model.as_deref(),
        system.as_deref(),
        &first_user,
    )?;
    let tools = normalize_tools(payload.tools);
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        payload.stream,
        payload.temperature,
        Some(max_tokens),
        tools,
        payload.tool_choice,
        None,
        extra,
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

            let deadline = tokio::time::Instant::now() + Duration::from_secs(stream_timeout);
            let mut timed_out = false;
            let mut next_block_index: usize = 0;
            let mut open_thinking_index: Option<usize> = None;
            let mut open_text_index: Option<usize> = None;
            let mut tool_blocks: BTreeMap<usize, usize> = BTreeMap::new();
            let mut emitted_any_block = false;
            let mut saw_tool_calls = false;

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
                        let StreamChunk {
                            delta,
                            thinking_delta,
                            signature_delta,
                            redacted_thinking,
                            tool_calls,
                            finish_reason,
                            usage,
                        } = chunk;

                        if let Some(redacted) = redacted_thinking {
                            if let Some(idx) = open_thinking_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            let idx = next_block_index;
                            next_block_index += 1;
                            emitted_any_block = true;
                            yield Ok(Event::default().event("content_block_start").data(
                                json!({
                                    "type": "content_block_start",
                                    "index": idx,
                                    "content_block": {"type": "redacted_thinking", "data": redacted},
                                })
                                .to_string(),
                            ));
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                        }

                        if thinking_delta.is_some() || signature_delta.is_some() {
                            let idx = match open_thinking_index {
                                Some(idx) => idx,
                                None => {
                                    if let Some(t_idx) = open_text_index.take() {
                                        yield Ok(Event::default()
                                            .event("content_block_stop")
                                            .data(json!({"type": "content_block_stop", "index": t_idx}).to_string()));
                                    }
                                    let idx = next_block_index;
                                    next_block_index += 1;
                                    open_thinking_index = Some(idx);
                                    emitted_any_block = true;
                                    let block_start = json!({
                                        "type": "content_block_start",
                                        "index": idx,
                                        "content_block": {"type": "thinking", "thinking": ""},
                                    });
                                    yield Ok(Event::default()
                                        .event("content_block_start")
                                        .data(block_start.to_string()));
                                    idx
                                }
                            };
                            if let Some(thinking) = thinking_delta
                                && !thinking.is_empty()
                            {
                                let payload = json!({
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "thinking_delta", "thinking": thinking},
                                });
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(payload.to_string()));
                            }
                            if let Some(sig) = signature_delta
                                && !sig.is_empty()
                            {
                                let payload = json!({
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "signature_delta", "signature": sig},
                                });
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(payload.to_string()));
                            }
                        }

                        if !delta.is_empty() {
                            if let Some(idx) = open_thinking_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            let idx = match open_text_index {
                                Some(idx) => idx,
                                None => {
                                    let idx = next_block_index;
                                    next_block_index += 1;
                                    open_text_index = Some(idx);
                                    emitted_any_block = true;
                                    let block_start = json!({
                                        "type": "content_block_start",
                                        "index": idx,
                                        "content_block": {"type": "text", "text": ""},
                                    });
                                    yield Ok(Event::default()
                                        .event("content_block_start")
                                        .data(block_start.to_string()));
                                    idx
                                }
                            };
                            let payload = json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": delta},
                            });
                            yield Ok(Event::default()
                                .event("content_block_delta")
                                .data(payload.to_string()));
                        }
                        if !tool_calls.is_empty() {
                            saw_tool_calls = true;
                            if let Some(idx) = open_thinking_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            if let Some(idx) = open_text_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            for tc in tool_calls {
                                let idx = match tool_blocks.get(&tc.index).copied() {
                                    Some(idx) => idx,
                                    None => {
                                        let idx = next_block_index;
                                        next_block_index += 1;
                                        tool_blocks.insert(tc.index, idx);
                                        emitted_any_block = true;
                                        let id = tc.id.unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple()));
                                        let name = tc.name.unwrap_or_else(|| "tool".into());
                                        let block_start = json!({
                                            "type": "content_block_start",
                                            "index": idx,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": id,
                                                "name": name,
                                                "input": {},
                                            },
                                        });
                                        yield Ok(Event::default()
                                            .event("content_block_start")
                                            .data(block_start.to_string()));
                                        idx
                                    }
                                };
                                if let Some(partial) = tc.arguments_delta
                                    && !partial.is_empty()
                                {
                                    let payload = json!({
                                        "type": "content_block_delta",
                                        "index": idx,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": partial,
                                        },
                                    });
                                    yield Ok(Event::default()
                                        .event("content_block_delta")
                                        .data(payload.to_string()));
                                }
                            }
                        }
                        if let Some(reason) = finish_reason {
                            if let Some(idx) = open_thinking_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            if let Some(idx) = open_text_index.take() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            for idx in tool_blocks.values() {
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                            }
                            if !emitted_any_block {
                                yield Ok(Event::default().event("content_block_start").data(
                                    json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}).to_string()
                                ));
                                yield Ok(Event::default()
                                    .event("content_block_stop")
                                    .data(json!({"type": "content_block_stop", "index": 0}).to_string()));
                            }
                            let output_tokens = usage.map(|usage| usage.completion_tokens).unwrap_or(0);
                            let effective_reason = if saw_tool_calls && reason == "stop" {
                                "tool_use"
                            } else {
                                anthropic_stop_reason(&reason)
                            };
                            let delta = json!({
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": effective_reason,
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
                        if let Some(idx) = open_thinking_index.take() {
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                        }
                        if let Some(idx) = open_text_index.take() {
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                        }
                        for idx in tool_blocks.values() {
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": idx}).to_string()));
                        }
                        if !emitted_any_block {
                            yield Ok(Event::default().event("content_block_start").data(
                                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}).to_string()
                            ));
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json!({"type": "content_block_stop", "index": 0}).to_string()));
                        }
                        let stop_reason = if saw_tool_calls { "tool_use" } else { "end_turn" };
                        let delta = json!({
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
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
    let mut content_blocks = Vec::new();
    for block in &completion.thinking {
        match block {
            ThinkingBlock::Thinking {
                thinking,
                signature,
            } => {
                let mut obj = Map::new();
                obj.insert("type".into(), json!("thinking"));
                obj.insert("thinking".into(), json!(thinking));
                if let Some(sig) = signature {
                    obj.insert("signature".into(), json!(sig));
                }
                content_blocks.push(Value::Object(obj));
            }
            ThinkingBlock::RedactedThinking { data } => {
                content_blocks.push(json!({
                    "type": "redacted_thinking",
                    "data": data,
                }));
            }
        }
    }
    if !completion.content.is_empty()
        || (completion.tool_calls.is_empty() && completion.thinking.is_empty())
    {
        content_blocks.push(json!({"type": "text", "text": completion.content}));
    }
    for call in &completion.tool_calls {
        let input: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
        content_blocks.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": input,
        }));
    }
    let stop_reason = if !completion.tool_calls.is_empty() && completion.finish_reason == "stop" {
        "tool_use"
    } else {
        anthropic_stop_reason(&completion.finish_reason)
    };
    let body = json!({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": context.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
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

fn validate_thinking_config(thinking: &Value) -> Result<()> {
    let Some(obj) = thinking.as_object() else {
        return Err(GatewayError::BadRequest(
            "thinking must be a JSON object".into(),
        ));
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("enabled") => {
            if let Some(budget) = obj.get("budget_tokens").and_then(Value::as_u64)
                && budget == 0
            {
                return Err(GatewayError::BadRequest(
                    "thinking.budget_tokens must be greater than zero".into(),
                ));
            }
            Ok(())
        }
        Some("disabled") | None => Ok(()),
        Some(other) => Err(GatewayError::BadRequest(format!(
            "unsupported thinking.type {other:?}"
        ))),
    }
}

fn normalize_messages(
    system: Option<Value>,
    raw: Vec<AnthropicMessage>,
) -> (Option<String>, Vec<ChatMessage>) {
    let mut system_parts = Vec::new();
    if let Some(system) = system {
        let text = parse_json_content(&system).as_plain_text();
        if !text.trim().is_empty() {
            system_parts.push(text);
        }
    }
    let mut messages = Vec::with_capacity(raw.len());
    for message in raw {
        let role = message.role.trim();
        let role = if role.is_empty() { "user" } else { role };
        if let Some(Value::Array(parts)) = &message.content
            && parts.iter().any(|part| {
                matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("tool_use" | "tool_result")
                )
            })
        {
            let mut rich_parts: Vec<ContentPart> = Vec::new();
            let mut tool_calls = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        if let Some(name) = part.get("name").and_then(Value::as_str) {
                            let id = part
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToString::to_string)
                                .unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple()));
                            let arguments = part
                                .get("input")
                                .map(Value::to_string)
                                .unwrap_or_else(|| "{}".to_string());
                            tool_calls.push(ToolCall {
                                id,
                                name: name.to_string(),
                                arguments,
                            });
                        }
                    }
                    Some("tool_result") => {
                        let call_id = part
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .map(ToString::to_string);
                        let result_text = part
                            .get("content")
                            .map(parse_json_content)
                            .unwrap_or_default();
                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: result_text,
                            tool_calls: Vec::new(),
                            tool_call_id: call_id,
                            name: None,
                        });
                    }
                    Some("thinking" | "redacted_thinking") => {}
                    _ => match parse_json_content(part) {
                        MessageContent::Parts(mut inner) => rich_parts.append(&mut inner),
                        MessageContent::Text(t) if !t.trim().is_empty() => {
                            rich_parts.push(ContentPart::Text {
                                text: t,
                                cache_control: part.get("cache_control").cloned(),
                            });
                        }
                        _ => {}
                    },
                }
            }
            if !rich_parts.is_empty() || !tool_calls.is_empty() {
                messages.push(ChatMessage {
                    role: role.to_string(),
                    content: if rich_parts.is_empty() {
                        MessageContent::Text(String::new())
                    } else {
                        MessageContent::Parts(rich_parts)
                    },
                    tool_calls,
                    tool_call_id: None,
                    name: None,
                });
            }
            continue;
        }
        let content = message
            .content
            .as_ref()
            .map(parse_json_content)
            .unwrap_or_default();
        if role == "system" {
            let text = content.as_plain_text();
            if !text.trim().is_empty() {
                system_parts.push(text);
            }
        } else {
            messages.push(ChatMessage {
                role: role.to_string(),
                content,
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
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
