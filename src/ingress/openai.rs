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
    driver::{
        ChatMessage, MessageContent, ResponseFormat, StreamChunk, ThinkingBlock, ToolCall,
        ToolCallDelta, Usage, tool_bridge::normalize_tools, truncate,
    },
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, StreamUsageGuard, completion, execute_or_cached, execution_request,
        insert_routing_headers, into_chunk_stream, parse_json_content, prepare_routed,
        record_completion_outcome, record_error_outcome, sse_response, start_ingress_span,
        unix_now,
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
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
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
    let response_format = parse_openai_response_format(payload.response_format)?;
    let tools = normalize_tools(payload.tools);
    let context = prepare_routed(
        &state,
        &headers,
        payload.model.as_deref(),
        system.as_deref(),
        &messages,
        &tools,
    )?;
    let started = std::time::Instant::now();
    let ingress_span = start_ingress_span(&state, &context, "/v1/chat/completions");
    let max_tokens = payload.max_tokens.or(payload.max_completion_tokens);
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        payload.stream,
        payload.temperature,
        max_tokens,
        tools,
        payload.tool_choice,
        response_format,
        payload.extra,
        cancel.clone(),
    );
    let dispatched = match execute_or_cached(&state, &context, &headers, request).await {
        Ok(dispatched) => dispatched,
        Err(error) => {
            record_error_outcome(
                &state,
                &context,
                None,
                started.elapsed(),
                &error.to_string(),
                Some(ingress_span),
            );
            return Err(error);
        }
    };
    let backend = dispatched.routed.backend_id;
    let fallback = dispatched.routed.fallback_used;
    let driver_response = dispatched.routed.response;
    let cache_status = dispatched.cache_status;
    let cache_miss_key = dispatched.cache_miss_key;

    if payload.stream {
        let config = state.config();
        let keep_alive = config.server.keep_alive_interval_secs;
        let stream_timeout = config.server.stream_timeout_secs.max(1);
        let mut chunks = into_chunk_stream(driver_response);
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let model = context.model.clone();
        let request_id = context.request_id.clone();
        let stream_cancel = cancel.clone();
        let stream_backend = backend.clone();
        let mut usage_guard = StreamUsageGuard::new(
            state.clone(),
            context,
            stream_backend,
            fallback,
            started,
            Some(ingress_span),
            cache_miss_key,
        );

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
                    usage_guard.fail("stream timeout");
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
                        usage_guard.observe_chunk(&chunk);
                        let StreamChunk {
                            delta,
                            thinking_delta,
                            tool_calls,
                            finish_reason,
                            usage,
                            ..
                        } = chunk;
                        if let Some(reasoning) = thinking_delta
                            && !reasoning.is_empty()
                        {
                            let payload = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"reasoning_content": reasoning},
                                    "finish_reason": null,
                                }],
                            });
                            yield Ok(Event::default().data(payload.to_string()));
                        }
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
                        if !tool_calls.is_empty() {
                            saw_tool_calls = true;
                            let deltas: Vec<Value> = tool_calls.iter().map(tool_call_delta_json).collect();
                            let payload = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"tool_calls": deltas},
                                    "finish_reason": null,
                                }],
                            });
                            yield Ok(Event::default().data(payload.to_string()));
                        }
                        if let Some(reason) = finish_reason {
                            let effective_reason = if saw_tool_calls && reason == "stop" {
                                "tool_calls".to_string()
                            } else {
                                reason
                            };
                            let _ = usage_guard.complete(usage, &effective_reason);
                            let mut payload = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": effective_reason,
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
                        usage_guard.fail(&error.to_string());
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
                        let effective_reason = if saw_tool_calls { "tool_calls" } else { "stop" };
                        let _ = usage_guard.complete(None, effective_reason);
                        let payload = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": effective_reason,
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
        insert_routing_headers(&mut response, &backend, fallback, &request_id, cache_status);
        return Ok(response);
    }

    let completion = match completion(driver_response).await {
        Ok(c) => {
            record_completion_outcome(
                &state,
                &context,
                &backend,
                fallback,
                c.usage,
                started.elapsed(),
                Some(ingress_span),
            );
            if let Some(key) = cache_miss_key {
                state.cache().store(key, &backend, &c);
            }
            c
        }
        Err(error) => {
            record_error_outcome(
                &state,
                &context,
                Some(&backend),
                started.elapsed(),
                &error.to_string(),
                Some(ingress_span),
            );
            return Err(error);
        }
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let mut message = json!({
        "role": "assistant",
        "content": if completion.content.is_empty() && !completion.tool_calls.is_empty() {
            Value::Null
        } else {
            json!(completion.content)
        },
    });
    if !completion.thinking.is_empty()
        && let Some(object) = message.as_object_mut()
    {
        let reasoning: String = completion
            .thinking
            .iter()
            .filter_map(|block| match block {
                ThinkingBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                ThinkingBlock::RedactedThinking { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !reasoning.is_empty() {
            object.insert("reasoning_content".into(), json!(reasoning));
        }
    }
    if !completion.tool_calls.is_empty()
        && let Some(object) = message.as_object_mut()
    {
        let calls: Vec<Value> = completion.tool_calls.iter().map(tool_call_json).collect();
        object.insert("tool_calls".into(), Value::Array(calls));
    }
    let finish_reason = if !completion.tool_calls.is_empty() && completion.finish_reason == "stop" {
        "tool_calls".to_string()
    } else {
        completion.finish_reason
    };
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_now(),
        "model": context.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage_json(&completion.usage),
    });
    let mut response = (StatusCode::OK, Json(body)).into_response();
    insert_routing_headers(
        &mut response,
        &backend,
        fallback,
        &context.request_id,
        cache_status,
    );
    Ok(response)
}

pub(crate) fn parse_openai_response_format(value: Option<Value>) -> Result<Option<ResponseFormat>> {
    let Some(val) = value else {
        return Ok(None);
    };
    let kind = val
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| GatewayError::BadRequest("response_format.type must be a string".into()))?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(ResponseFormat::JsonObject)),
        "json_schema" => {
            let js = val.get("json_schema").unwrap_or(&val);
            let schema = js.get("schema").cloned().ok_or_else(|| {
                GatewayError::BadRequest("response_format.json_schema.schema is required".into())
            })?;
            let name = js
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let strict = js.get("strict").and_then(Value::as_bool).unwrap_or(false);
            Ok(Some(ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
                raw: val,
            }))
        }
        other => Err(GatewayError::BadRequest(format!(
            "unsupported response_format.type {other:?}"
        ))),
    }
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
            .map(parse_json_content)
            .unwrap_or_default();
        if role == "system" || role == "developer" {
            let text = content.as_plain_text();
            if !text.trim().is_empty() {
                system_parts.push(text);
            }
        } else {
            let tool_calls = parse_openai_tool_calls(&message.tool_calls);
            let content = if role == "tool" {
                MessageContent::Text(content.as_plain_text())
            } else {
                content
            };
            messages.push(ChatMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: message.tool_call_id,
                name: message.name,
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

fn parse_openai_tool_calls(raw: &[Value]) -> Vec<ToolCall> {
    raw.iter()
        .filter_map(|call| {
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
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
        .collect()
}

pub(crate) fn tool_call_json(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        }
    })
}

fn tool_call_delta_json(delta: &ToolCallDelta) -> Value {
    let mut item = Map::new();
    item.insert("index".into(), json!(delta.index));
    if let Some(id) = &delta.id {
        item.insert("id".into(), json!(id));
        item.insert("type".into(), json!("function"));
    }
    let mut function = Map::new();
    if let Some(name) = &delta.name {
        function.insert("name".into(), json!(name));
    }
    if let Some(args) = &delta.arguments_delta {
        function.insert("arguments".into(), json!(args));
    }
    if !function.is_empty() {
        item.insert("function".into(), Value::Object(function));
    }
    Value::Object(item)
}

pub(crate) fn usage_json(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    })
}
