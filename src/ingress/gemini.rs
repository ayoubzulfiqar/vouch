//! SPDX-License-Identifier: GPL-3.0-or-later
//! Google Gemini API ingress (`POST /v1beta/models/{model}:{generateContent,streamGenerateContent}`).

use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response, sse::Event},
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    driver::{
        ChatMessage, ContentPart, MessageContent, ResponseFormat, StreamChunk, ThinkingBlock,
        ToolCall, Usage, attachments::parse_data_uri, tool_bridge::normalize_tool_definition,
        truncate,
    },
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, StreamUsageGuard, completion, execute_or_cached, execution_request,
        insert_dispatched_headers, insert_handoff_headers, insert_routing_headers,
        into_chunk_stream, prepare_routed_with_key_hint, record_completion_outcome,
        record_error_outcome, record_memory_turn, sse_response, start_ingress_span,
    },
    state::AppState,
    telemetry::redact_text,
};

/// Query parameters accepted by Gemini endpoints (`?alt=sse` and `?key=<token>`).
#[derive(Debug, Default, Deserialize)]
pub struct GeminiQuery {
    #[serde(default)]
    alt: Option<String>,
    #[serde(default)]
    key: Option<String>,
}

/// Google Gemini `generateContent` / `streamGenerateContent` request body.
#[derive(Debug, Deserialize)]
pub struct GeminiGenerateRequest {
    #[serde(default)]
    contents: Vec<GeminiContent>,
    #[serde(default, rename = "systemInstruction", alias = "system_instruction")]
    system_instruction: Option<GeminiContent>,
    #[serde(default)]
    tools: Vec<GeminiTool>,
    #[serde(default, rename = "generationConfig", alias = "generation_config")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    parts: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiTool {
    #[serde(
        default,
        rename = "functionDeclarations",
        alias = "function_declarations"
    )]
    function_declarations: Vec<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct GeminiGenerationConfig {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default, rename = "maxOutputTokens", alias = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(default, rename = "responseMimeType", alias = "response_mime_type")]
    response_mime_type: Option<String>,
    #[serde(default, rename = "responseSchema", alias = "response_schema")]
    response_schema: Option<Value>,
}

/// Handle `POST /v1beta/models/{*model_action}`.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(model_action): Path<String>,
    Query(query): Query<GeminiQuery>,
    payload: std::result::Result<Json<GeminiGenerateRequest>, JsonRejection>,
) -> Response {
    let trimmed = model_action.trim_start_matches('/');
    let Some((raw_model, action)) = trimmed.rsplit_once(':') else {
        return into_gemini_error_response(GatewayError::NotFound(format!(
            "unsupported Gemini model path {model_action:?}; expected <model>:generateContent or <model>:streamGenerateContent"
        )));
    };
    let stream = match action {
        "generateContent" => false,
        "streamGenerateContent" => true,
        _ => {
            return into_gemini_error_response(GatewayError::NotFound(format!(
                "unsupported Gemini action {action:?}"
            )));
        }
    };

    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(rejection) => {
            return into_gemini_error_response(GatewayError::BadRequest(format!(
                "invalid Gemini request body: {}",
                truncate(&rejection.body_text(), 500)
            )));
        }
    };

    match generate_inner(
        state,
        headers,
        raw_model.to_string(),
        stream,
        query,
        payload,
    )
    .await
    {
        Ok(resp) => resp,
        Err(error) => into_gemini_error_response(error),
    }
}

async fn generate_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    raw_model: String,
    stream: bool,
    query: GeminiQuery,
    payload: GeminiGenerateRequest,
) -> Result<Response> {
    let system = payload
        .system_instruction
        .as_ref()
        .map(extract_plain_parts_text)
        .filter(|s| !s.trim().is_empty());
    let messages = normalize_contents(payload.contents);
    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "contents must contain at least one non-empty part".into(),
        ));
    }

    let mut tools = Vec::new();
    for tool_group in payload.tools {
        for decl in tool_group.function_declarations {
            tools.push(normalize_tool_definition(&decl));
        }
    }

    let gen_cfg = payload.generation_config.unwrap_or_default();
    let response_format = build_response_format(&gen_cfg);

    let context = prepare_routed_with_key_hint(
        &state,
        &headers,
        query.key.as_deref(),
        Some(&raw_model),
        system.as_deref(),
        &messages,
        &tools,
    )?;

    let started = Instant::now();
    let ingress_span = start_ingress_span(&state, &context, "/v1beta/models");
    let cancel = CancellationToken::new();
    let request = execution_request(
        &context,
        system,
        messages,
        stream,
        gen_cfg.temperature,
        gen_cfg.max_output_tokens,
        tools,
        None,
        response_format,
        payload.extra,
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
    let backend = dispatched.routed.backend_id.clone();
    let fallback = dispatched.routed.fallback_used;
    let driver_response = dispatched.routed.response;
    let cache_status = dispatched.cache_status;
    let cache_miss_key = dispatched.cache_miss_key;
    let coalesced = dispatched.coalesced;
    let request_id = context.request_id.clone();
    let handoff_from = context.handoff_from();

    if stream {
        let sse_mode = query
            .alt
            .as_deref()
            .is_some_and(|v| v.eq_ignore_ascii_case("sse"));
        let config = state.config();
        let keep_alive = config.server.keep_alive_interval_secs;
        let stream_timeout = config.server.stream_timeout_secs.max(1);
        let mut chunks = into_chunk_stream(driver_response);
        let model = context.model.clone();
        let mut usage_guard = StreamUsageGuard::new(
            state.clone(),
            context,
            backend.clone(),
            fallback,
            started,
            Some(ingress_span),
            cache_miss_key,
        );

        let (tx, rx) = mpsc::channel::<Result<Value>>(32);
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            let _cancel_guard = CancelOnDrop(cancel_clone.clone());
            let deadline = tokio::time::Instant::now() + Duration::from_secs(stream_timeout);
            let mut saw_tool_calls = false;
            let mut tool_calls: BTreeMap<usize, (ToolCall, bool)> = BTreeMap::new();
            let mut final_usage: Option<Usage> = None;

            loop {
                let item = tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => {
                        cancel_clone.cancel();
                        usage_guard.fail("stream timeout");
                        let _ = tx.send(Err(GatewayError::Timeout(Duration::from_secs(stream_timeout)))).await;
                        break;
                    }
                    item = chunks.next() => item,
                };
                match item {
                    Some(Ok(chunk)) => {
                        usage_guard.observe_chunk(&chunk);
                        let StreamChunk {
                            delta,
                            thinking_delta,
                            tool_calls: tc_deltas,
                            finish_reason,
                            usage,
                            ..
                        } = chunk;
                        if let Some(u) = usage {
                            final_usage = Some(u);
                        }
                        if let Some(thinking) = thinking_delta.as_deref()
                            && !thinking.is_empty()
                        {
                            let body = gemini_chunk_json(
                                &model,
                                "",
                                &[],
                                &[ThinkingBlock::Thinking {
                                    thinking: thinking.to_string(),
                                    signature: None,
                                }],
                                None,
                                None,
                            );
                            if tx.send(Ok(body)).await.is_err() {
                                break;
                            }
                        }
                        if !delta.is_empty() {
                            let body = gemini_chunk_json(&model, &delta, &[], &[], None, None);
                            if tx.send(Ok(body)).await.is_err() {
                                break;
                            }
                        }
                        if !tc_deltas.is_empty() {
                            saw_tool_calls = true;
                            let mut ready_now = Vec::new();
                            for td in tc_deltas {
                                let entry = tool_calls.entry(td.index).or_insert_with(|| {
                                    (
                                        ToolCall {
                                            id: td.id.clone().unwrap_or_else(|| {
                                                format!("call_{}", Uuid::new_v4().simple())
                                            }),
                                            name: td.name.clone().unwrap_or_default(),
                                            arguments: String::new(),
                                        },
                                        false,
                                    )
                                });
                                if let Some(id) = td.id {
                                    entry.0.id = id;
                                }
                                if let Some(name) = td.name {
                                    entry.0.name = name;
                                }
                                if let Some(args) = td.arguments_delta {
                                    entry.0.arguments.push_str(&args);
                                }
                                if !entry.1
                                    && !entry.0.name.is_empty()
                                    && serde_json::from_str::<Value>(&entry.0.arguments).is_ok()
                                {
                                    entry.1 = true;
                                    ready_now.push(entry.0.clone());
                                }
                            }
                            if !ready_now.is_empty() {
                                let body =
                                    gemini_chunk_json(&model, "", &ready_now, &[], None, None);
                                if tx.send(Ok(body)).await.is_err() {
                                    break;
                                }
                            }
                        }
                        if let Some(reason) = finish_reason {
                            let remaining_calls: Vec<ToolCall> = tool_calls
                                .into_values()
                                .filter_map(|(call, emitted)| (!emitted).then_some(call))
                                .collect();
                            let effective = usage_guard.complete(final_usage.or(usage), &reason);
                            let gemini_reason = gemini_finish_reason(
                                &reason,
                                saw_tool_calls || !remaining_calls.is_empty(),
                            );
                            let body = gemini_chunk_json(
                                &model,
                                "",
                                &remaining_calls,
                                &[],
                                Some(gemini_reason),
                                Some(effective),
                            );
                            let _ = tx.send(Ok(body)).await;
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        usage_guard.fail(&error.to_string());
                        let _ = tx.send(Err(error)).await;
                        break;
                    }
                    None => {
                        let remaining_calls: Vec<ToolCall> = tool_calls
                            .into_values()
                            .filter_map(|(call, emitted)| (!emitted).then_some(call))
                            .collect();
                        let reason = if saw_tool_calls || !remaining_calls.is_empty() {
                            "tool_calls"
                        } else {
                            "stop"
                        };
                        let effective = usage_guard.complete(final_usage, reason);
                        let gemini_reason = gemini_finish_reason(
                            reason,
                            saw_tool_calls || !remaining_calls.is_empty(),
                        );
                        let body = gemini_chunk_json(
                            &model,
                            "",
                            &remaining_calls,
                            &[],
                            Some(gemini_reason),
                            Some(effective),
                        );
                        let _ = tx.send(Ok(body)).await;
                        break;
                    }
                }
            }
        });

        let mut response = if sse_mode {
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|item| {
                let evt = match item {
                    Ok(val) => Event::default().data(val.to_string()),
                    Err(err) => Event::default().data(gemini_error_json(&err).to_string()),
                };
                Ok::<Event, Infallible>(evt)
            });
            sse_response(stream, keep_alive)
        } else {
            // Chunked JSON array (`[chunk1, chunk2]`) matching non-SSE Gemini streamGenerateContent.
            let mut rx = rx;
            let json_stream = async_stream::stream! {
                let mut first = true;
                yield Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b"["));
                while let Some(item) = rx.recv().await {
                    let line = match item {
                        Ok(val) => val.to_string(),
                        Err(err) => gemini_error_json(&err).to_string(),
                    };
                    if !first {
                        yield Ok(axum::body::Bytes::from_static(b","));
                    }
                    first = false;
                    yield Ok(axum::body::Bytes::from(line));
                }
                yield Ok(axum::body::Bytes::from_static(b"]"));
            };
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from_stream(json_stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        };
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

    let finish = gemini_finish_reason(&completion.finish_reason, !completion.tool_calls.is_empty());
    let body = gemini_chunk_json(
        &context.model,
        &completion.content,
        &completion.tool_calls,
        &completion.thinking,
        Some(finish),
        Some(completion.usage),
    );
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

fn extract_plain_parts_text(content: &GeminiContent) -> String {
    content
        .parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_contents(contents: Vec<GeminiContent>) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    for item in contents {
        let role = match item.role.as_deref() {
            Some("model") | Some("assistant") => "assistant",
            _ => "user",
        };
        let mut parts: Vec<ContentPart> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut tool_responses: Vec<ChatMessage> = Vec::new();

        for part in item.parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let is_thought = part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if !is_thought {
                    parts.push(ContentPart::Text {
                        text: text.to_string(),
                        cache_control: None,
                    });
                }
            } else if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data"))
                && let Some(data) = inline.get("data").and_then(Value::as_str)
            {
                let mime = inline
                    .get("mimeType")
                    .or_else(|| inline.get("mime_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("image/png")
                    .to_string();
                parts.push(ContentPart::ImageBase64 {
                    media_type: mime,
                    data: data.to_string(),
                });
            } else if let Some(file) = part.get("fileData").or_else(|| part.get("file_data"))
                && let Some(uri) = file
                    .get("fileUri")
                    .or_else(|| file.get("file_uri"))
                    .and_then(Value::as_str)
            {
                if let Some((media_type, data)) = parse_data_uri(uri) {
                    parts.push(ContentPart::ImageBase64 { media_type, data });
                } else {
                    parts.push(ContentPart::ImageUrl {
                        url: uri.to_string(),
                    });
                }
            } else if let Some(fc) = part
                .get("functionCall")
                .or_else(|| part.get("function_call"))
                && let Some(name) = fc.get("name").and_then(Value::as_str)
            {
                let args = fc
                    .get("args")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "{}".into());
                let id = fc
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                calls.push(ToolCall {
                    id,
                    name: name.to_string(),
                    arguments: args,
                });
            } else if let Some(fr) = part
                .get("functionResponse")
                .or_else(|| part.get("function_response"))
            {
                let name = fr
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                let tool_call_id = fr
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .or_else(|| name.clone());
                let content_str = fr
                    .get("response")
                    .map(|r| match r {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                tool_responses.push(ChatMessage {
                    role: "tool".into(),
                    content: MessageContent::Text(content_str),
                    tool_calls: Vec::new(),
                    tool_call_id,
                    name,
                });
            }
        }

        if !parts.is_empty() || !calls.is_empty() {
            let content = if parts.iter().any(|p| !matches!(p, ContentPart::Text { .. })) {
                MessageContent::Parts(parts)
            } else {
                let text = parts
                    .into_iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text, .. } => Some(text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                MessageContent::Text(text)
            };
            out.push(ChatMessage {
                role: role.to_string(),
                content,
                tool_calls: calls,
                tool_call_id: None,
                name: None,
            });
        }
        out.extend(tool_responses);
    }
    out
}

fn build_response_format(cfg: &GeminiGenerationConfig) -> Option<ResponseFormat> {
    if let Some(schema) = &cfg.response_schema {
        return Some(ResponseFormat::JsonSchema {
            name: Some("gemini_response".into()),
            schema: schema.clone(),
            strict: true,
            raw: json!({
                "type": "json_schema",
                "json_schema": { "name": "gemini_response", "schema": schema, "strict": true }
            }),
        });
    }
    if cfg
        .response_mime_type
        .as_deref()
        .is_some_and(|m| m.eq_ignore_ascii_case("application/json"))
    {
        return Some(ResponseFormat::JsonObject);
    }
    None
}

fn gemini_finish_reason(reason: &str, has_tool_calls: bool) -> &'static str {
    if has_tool_calls && reason == "stop" {
        return "STOP";
    }
    match reason {
        "length" | "max_tokens" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

fn gemini_chunk_json(
    model: &str,
    text: &str,
    tool_calls: &[ToolCall],
    thinking: &[ThinkingBlock],
    finish_reason: Option<&str>,
    usage: Option<Usage>,
) -> Value {
    let mut parts = Vec::new();
    for block in thinking {
        if let ThinkingBlock::Thinking { thinking, .. } = block
            && !thinking.is_empty()
        {
            parts.push(json!({ "thought": true, "text": thinking }));
        }
    }
    if !text.is_empty() {
        parts.push(json!({ "text": text }));
    }
    for call in tool_calls {
        let args: Value = serde_json::from_str(&call.arguments)
            .unwrap_or_else(|_| json!({ "raw": call.arguments }));
        parts.push(json!({
            "functionCall": {
                "name": call.name,
                "args": args,
            }
        }));
    }
    if parts.is_empty() {
        parts.push(json!({ "text": "" }));
    }
    let mut candidate = Map::new();
    candidate.insert(
        "content".into(),
        json!({
            "role": "model",
            "parts": parts,
        }),
    );
    candidate.insert("index".into(), json!(0));
    if let Some(reason) = finish_reason {
        candidate.insert("finishReason".into(), json!(reason));
    }
    let mut obj = Map::new();
    obj.insert(
        "candidates".into(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    obj.insert("modelVersion".into(), json!(model));
    if let Some(u) = usage {
        obj.insert(
            "usageMetadata".into(),
            json!({
                "promptTokenCount": u.prompt_tokens,
                "candidatesTokenCount": u.completion_tokens,
                "totalTokenCount": u.total_tokens,
            }),
        );
    }
    Value::Object(obj)
}

fn gemini_error_json(error: &GatewayError) -> Value {
    let code = error.status().as_u16();
    let status = match error.status() {
        StatusCode::BAD_REQUEST | StatusCode::UNSUPPORTED_MEDIA_TYPE => "INVALID_ARGUMENT",
        StatusCode::UNAUTHORIZED => "UNAUTHENTICATED",
        StatusCode::FORBIDDEN => "PERMISSION_DENIED",
        StatusCode::NOT_FOUND => "NOT_FOUND",
        StatusCode::TOO_MANY_REQUESTS => "RESOURCE_EXHAUSTED",
        StatusCode::SERVICE_UNAVAILABLE => "UNAVAILABLE",
        StatusCode::GATEWAY_TIMEOUT | StatusCode::REQUEST_TIMEOUT => "DEADLINE_EXCEEDED",
        _ => "INTERNAL",
    };
    json!({
        "error": {
            "code": code,
            "message": redact_text(&error.to_string()),
            "status": status,
        }
    })
}

pub(crate) fn into_gemini_error_response(error: GatewayError) -> Response {
    let status = error.status();
    let body = gemini_error_json(&error);
    (status, Json(body)).into_response()
}
