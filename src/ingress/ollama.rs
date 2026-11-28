//! SPDX-License-Identifier: GPL-3.0-or-later
//! Ollama API ingress (`GET /api/tags`, `POST /api/chat`, `POST /api/generate`).

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json,
    body::Body,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    driver::{
        ChatMessage, ContentPart, MessageContent, ResponseFormat, StreamChunk, ToolCall, Usage,
        attachments::parse_data_uri, tool_bridge::normalize_tools, truncate,
    },
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, RequestContext, StreamUsageGuard, completion, execute_or_cached,
        execution_request, insert_dispatched_headers, insert_handoff_headers,
        insert_routing_headers, into_chunk_stream, iso8601_utc_now, prepare_routed,
        record_completion_outcome, record_error_outcome, record_memory_turn, start_ingress_span,
    },
    state::AppState,
    telemetry::redact_text,
};

/// Ollama `/api/chat` request payload.
#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default = "default_ollama_stream")]
    stream: bool,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    options: Option<OllamaOptions>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

/// Ollama `/api/generate` request payload.
#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default = "default_ollama_stream")]
    stream: bool,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    options: Option<OllamaOptions>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    tool_calls: Vec<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaOptions {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    num_predict: Option<u32>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

fn default_ollama_stream() -> bool {
    true
}

/// Handle `GET /api/tags`.
pub async fn tags(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(error) = state.tokens().authenticate(&headers) {
        return into_ollama_error_response(error);
    }
    let models: Vec<Value> = state
        .router()
        .models()
        .into_iter()
        .map(|name| {
            json!({
                "name": name,
                "model": name,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": 0,
                "digest": "vouch",
                "details": {
                    "format": "vouch",
                    "family": "vouch",
                }
            })
        })
        .collect();
    Json(json!({ "models": models })).into_response()
}

/// Handle `POST /api/chat`.
pub async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(p) => p,
        Err(rejection) => {
            return into_ollama_error_response(GatewayError::BadRequest(format!(
                "invalid Ollama chat request body: {}",
                truncate(&rejection.body_text(), 500)
            )));
        }
    };
    match chat_inner(state, headers, payload).await {
        Ok(r) => r,
        Err(error) => into_ollama_error_response(error),
    }
}

/// Handle `POST /api/generate`.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<OllamaGenerateRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(p) => p,
        Err(rejection) => {
            return into_ollama_error_response(GatewayError::BadRequest(format!(
                "invalid Ollama generate request body: {}",
                truncate(&rejection.body_text(), 500)
            )));
        }
    };
    match generate_inner(state, headers, payload).await {
        Ok(r) => r,
        Err(error) => into_ollama_error_response(error),
    }
}

async fn chat_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: OllamaChatRequest,
) -> Result<Response> {
    let mut system_parts = Vec::new();
    let mut messages = Vec::new();
    for msg in payload.messages {
        if msg.role.eq_ignore_ascii_case("system") {
            if !msg.content.trim().is_empty() {
                system_parts.push(msg.content);
            }
            continue;
        }
        let role = if msg.role.trim().is_empty() {
            "user".to_string()
        } else {
            msg.role
        };
        let content = build_ollama_content(&msg.content, &msg.images);
        let tool_calls = msg
            .tool_calls
            .into_iter()
            .filter_map(|tc| {
                let func = tc.get("function").unwrap_or(&tc);
                let name = func.get("name").and_then(Value::as_str)?.to_string();
                let args_val = func.get("arguments").cloned().unwrap_or_else(|| json!({}));
                let arguments = match args_val {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                let id = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();
        let tool_call_id = msg.tool_call_id.or_else(|| {
            if role.eq_ignore_ascii_case("tool") {
                msg.name.clone()
            } else {
                None
            }
        });
        messages.push(ChatMessage {
            role,
            content,
            tool_calls,
            tool_call_id,
            name: msg.name,
        });
    }
    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "messages must contain at least one non-system turn".into(),
        ));
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    let tools = normalize_tools(payload.tools);
    let response_format = parse_ollama_format(payload.format);
    let options = payload.options.unwrap_or_default();
    let mut extra = payload.extra;
    if !options.extra.is_empty() {
        extra.insert(
            "ollama_options".into(),
            Value::Object(options.extra.clone()),
        );
    }

    dispatch_ollama(
        state,
        headers,
        payload.model,
        system,
        messages,
        payload.stream,
        options,
        tools,
        response_format,
        extra,
        false,
        "/api/chat",
    )
    .await
}

async fn generate_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: OllamaGenerateRequest,
) -> Result<Response> {
    if payload.prompt.trim().is_empty() && payload.images.is_empty() {
        return Err(GatewayError::BadRequest(
            "prompt or images must not be empty".into(),
        ));
    }
    let content = build_ollama_content(&payload.prompt, &payload.images);
    let messages = vec![ChatMessage {
        role: "user".into(),
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    }];
    let response_format = parse_ollama_format(payload.format);
    let options = payload.options.unwrap_or_default();

    dispatch_ollama(
        state,
        headers,
        payload.model,
        payload.system,
        messages,
        payload.stream,
        options,
        Vec::new(),
        response_format,
        payload.extra,
        true,
        "/api/generate",
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_ollama(
    state: Arc<AppState>,
    headers: HeaderMap,
    requested_model: Option<String>,
    system: Option<String>,
    messages: Vec<ChatMessage>,
    stream: bool,
    options: OllamaOptions,
    tools: Vec<Value>,
    response_format: Option<ResponseFormat>,
    extra: Map<String, Value>,
    is_generate_mode: bool,
    endpoint: &'static str,
) -> Result<Response> {
    let context = prepare_routed(
        &state,
        &headers,
        requested_model.as_deref(),
        system.as_deref(),
        &messages,
        &tools,
    )?;
    let started = Instant::now();
    let ingress_span = start_ingress_span(&state, &context, endpoint);
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        stream,
        options.temperature,
        options.num_predict,
        tools,
        None,
        response_format,
        extra,
        cancel.clone(),
    );

    let dispatched = match execute_or_cached(&state, &context, &headers, request).await {
        Ok(d) => d,
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
    let coalesced = dispatched.coalesced;
    let request_id = context.request_id.clone();
    let handoff_from = context.handoff_from();

    if stream {
        let resp = build_ollama_ndjson_stream(
            state,
            context,
            backend.clone(),
            fallback,
            started,
            ingress_span,
            cache_miss_key,
            driver_response,
            cancel,
            is_generate_mode,
        );
        let mut response = resp;
        insert_routing_headers(&mut response, &backend, fallback, &request_id, cache_status);
        insert_handoff_headers(&mut response, handoff_from.as_deref());
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
            record_memory_turn(
                &state,
                &context,
                &backend,
                c.usage,
                &c.content,
                &c.tool_calls,
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

    let body = if is_generate_mode {
        ollama_generate_frame(
            &context.model,
            &completion.content,
            true,
            Some(&completion.finish_reason),
            Some(completion.usage),
        )
    } else {
        ollama_chat_frame(
            &context.model,
            &completion.content,
            &completion.tool_calls,
            true,
            Some(&completion.finish_reason),
            Some(completion.usage),
        )
    };
    let mut response = Json(body).into_response();
    insert_dispatched_headers(
        &mut response,
        &backend,
        fallback,
        &request_id,
        cache_status,
        coalesced,
    );
    insert_handoff_headers(&mut response, handoff_from.as_deref());
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
fn build_ollama_ndjson_stream(
    state: Arc<AppState>,
    context: RequestContext,
    backend: String,
    fallback: bool,
    started: Instant,
    ingress_span: crate::telemetry::ActiveSpan,
    cache_miss_key: Option<String>,
    driver_response: crate::driver::DriverResponse,
    cancel: CancellationToken,
    is_generate_mode: bool,
) -> Response {
    let stream_timeout = state.config().server.stream_timeout_secs.max(1);
    let model = context.model.clone();
    let mut chunks = into_chunk_stream(driver_response);
    let mut usage_guard = StreamUsageGuard::new(
        state,
        context,
        backend,
        fallback,
        started,
        Some(ingress_span),
        cache_miss_key,
    );
    let (tx, mut rx) = mpsc::channel::<axum::body::Bytes>(32);
    let cancel_clone = cancel;

    tokio::spawn(async move {
        let _cancel_guard = CancelOnDrop(cancel_clone.clone());
        let deadline = tokio::time::Instant::now() + Duration::from_secs(stream_timeout);
        let mut final_usage: Option<Usage> = None;
        let mut pending_calls: BTreeMap<usize, ToolCall> = BTreeMap::new();

        loop {
            let item = tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    cancel_clone.cancel();
                    usage_guard.fail("stream timeout");
                    let err_line = format!("{}\n", json!({ "error": "stream timeout" }));
                    let _ = tx.send(axum::body::Bytes::from(err_line)).await;
                    break;
                }
                item = chunks.next() => item,
            };
            match item {
                Some(Ok(chunk)) => {
                    usage_guard.observe_chunk(&chunk);
                    let StreamChunk {
                        delta,
                        tool_calls,
                        finish_reason,
                        usage,
                        ..
                    } = chunk;
                    if let Some(u) = usage {
                        final_usage = Some(u);
                    }
                    for td in tool_calls {
                        let entry = pending_calls.entry(td.index).or_insert_with(|| ToolCall {
                            id: td
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple())),
                            name: td.name.clone().unwrap_or_default(),
                            arguments: String::new(),
                        });
                        if let Some(id) = td.id {
                            entry.id = id;
                        }
                        if let Some(name) = td.name {
                            entry.name = name;
                        }
                        if let Some(args) = td.arguments_delta {
                            entry.arguments.push_str(&args);
                        }
                    }
                    if !delta.is_empty() {
                        let frame = if is_generate_mode {
                            ollama_generate_frame(&model, &delta, false, None, None)
                        } else {
                            ollama_chat_frame(&model, &delta, &[], false, None, None)
                        };
                        let line = format!("{frame}\n");
                        if tx.send(axum::body::Bytes::from(line)).await.is_err() {
                            break;
                        }
                    }
                    if let Some(reason) = finish_reason {
                        let calls: Vec<ToolCall> = pending_calls.into_values().collect();
                        let effective = usage_guard.complete(final_usage.or(usage), &reason);
                        let frame = if is_generate_mode {
                            ollama_generate_frame(&model, "", true, Some(&reason), Some(effective))
                        } else {
                            ollama_chat_frame(
                                &model,
                                "",
                                &calls,
                                true,
                                Some(&reason),
                                Some(effective),
                            )
                        };
                        let _ = tx.send(axum::body::Bytes::from(format!("{frame}\n"))).await;
                        break;
                    }
                }
                Some(Err(error)) => {
                    usage_guard.fail(&error.to_string());
                    let err_line =
                        format!("{}\n", json!({ "error": redact_text(&error.to_string()) }));
                    let _ = tx.send(axum::body::Bytes::from(err_line)).await;
                    break;
                }
                None => {
                    let calls: Vec<ToolCall> = pending_calls.into_values().collect();
                    let effective = usage_guard.complete(final_usage, "stop");
                    let frame = if is_generate_mode {
                        ollama_generate_frame(&model, "", true, Some("stop"), Some(effective))
                    } else {
                        ollama_chat_frame(&model, "", &calls, true, Some("stop"), Some(effective))
                    };
                    let _ = tx.send(axum::body::Bytes::from(format!("{frame}\n"))).await;
                    break;
                }
            }
        }
    });

    let body_stream = async_stream::stream! {
        while let Some(bytes) = rx.recv().await {
            yield Ok::<_, std::io::Error>(bytes);
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn build_ollama_content(text: &str, images: &[String]) -> MessageContent {
    if images.is_empty() {
        return MessageContent::Text(text.to_string());
    }
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(ContentPart::Text {
            text: text.to_string(),
            cache_control: None,
        });
    }
    for raw_img in images {
        if let Some((media_type, data)) = parse_data_uri(raw_img) {
            parts.push(ContentPart::ImageBase64 { media_type, data });
        } else {
            parts.push(ContentPart::ImageBase64 {
                media_type: "image/png".into(),
                data: raw_img.clone(),
            });
        }
    }
    MessageContent::Parts(parts)
}

fn parse_ollama_format(format: Option<Value>) -> Option<ResponseFormat> {
    let val = format?;
    match &val {
        Value::String(s) if s.eq_ignore_ascii_case("json") => Some(ResponseFormat::JsonObject),
        Value::Object(_) => Some(ResponseFormat::JsonSchema {
            name: Some("ollama_format".into()),
            schema: val.clone(),
            strict: true,
            raw: json!({
                "type": "json_schema",
                "json_schema": { "name": "ollama_format", "schema": val, "strict": true }
            }),
        }),
        _ => None,
    }
}

fn ollama_chat_frame(
    model: &str,
    content: &str,
    tool_calls: &[ToolCall],
    done: bool,
    done_reason: Option<&str>,
    usage: Option<Usage>,
) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !tool_calls.is_empty()
        && let Some(obj) = message.as_object_mut()
    {
        let formatted: Vec<Value> = tool_calls
            .iter()
            .map(|c| {
                let args: Value = serde_json::from_str(&c.arguments)
                    .unwrap_or_else(|_| json!({ "raw": c.arguments }));
                json!({
                    "function": {
                        "name": c.name,
                        "arguments": args,
                    }
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(formatted));
    }
    let mut obj = Map::new();
    obj.insert("model".into(), json!(model));
    obj.insert("created_at".into(), json!(iso8601_utc_now()));
    obj.insert("message".into(), message);
    obj.insert("done".into(), json!(done));
    if done {
        obj.insert("done_reason".into(), json!(done_reason.unwrap_or("stop")));
        if let Some(u) = usage {
            obj.insert("prompt_eval_count".into(), json!(u.prompt_tokens));
            obj.insert("eval_count".into(), json!(u.completion_tokens));
        }
    }
    Value::Object(obj)
}

fn ollama_generate_frame(
    model: &str,
    response_text: &str,
    done: bool,
    done_reason: Option<&str>,
    usage: Option<Usage>,
) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(model));
    obj.insert("created_at".into(), json!(iso8601_utc_now()));
    obj.insert("response".into(), json!(response_text));
    obj.insert("done".into(), json!(done));
    if done {
        obj.insert("done_reason".into(), json!(done_reason.unwrap_or("stop")));
        if let Some(u) = usage {
            obj.insert("prompt_eval_count".into(), json!(u.prompt_tokens));
            obj.insert("eval_count".into(), json!(u.completion_tokens));
        }
    }
    Value::Object(obj)
}

pub(crate) fn into_ollama_error_response(error: GatewayError) -> Response {
    let status = error.status();
    let msg = redact_text(&error.to_string());
    (status, Json(json!({ "error": msg }))).into_response()
}
