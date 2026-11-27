//! SPDX-License-Identifier: GPL-3.0-or-later
//! OpenAI Responses API (`POST /v1/responses`) ingress adapter.
//!
//! Normalizes string or structured `input` items (including multimodal
//! `input_image` parts), `instructions`, `text.format` structured output schemas,
//! `tools`, and stateful `previous_response_id` / `store` chaining into an
//! [`ExecutionRequest`], dispatches through [`crate::router::Router`], and
//! emits standard Responses API JSON or SSE events.

use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::Event},
};
use dashmap::DashMap;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    driver::{
        ChatMessage, ContentPart, MessageContent, ResponseFormat, StreamChunk, ToolCall, Usage,
        content_to_text, tool_bridge::normalize_tools, truncate,
    },
    error::{GatewayError, Result},
    ingress::{
        CancelOnDrop, completion, execute, execution_request, first_user_text, into_chunk_stream,
        openai::parse_openai_response_format, parse_json_content, prepare, sse_response, unix_now,
    },
    state::AppState,
};

/// Persisted conversation state for stateful `/v1/responses` chaining.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    /// System prompt / instructions active at the end of that turn.
    pub system: Option<String>,
    /// Full message history including the assistant's output and tool calls.
    pub messages: Vec<ChatMessage>,
    stored_at: Instant,
}

/// Thread-safe in-memory store for `previous_response_id` chaining with TTL
/// eviction.
#[derive(Debug, Default)]
pub struct ResponseStore {
    entries: DashMap<String, StoredResponse>,
}

impl ResponseStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up an unexpired stored response.
    pub fn get(&self, id: &str, ttl: Duration) -> Option<StoredResponse> {
        let entry = self.entries.get(id)?;
        if entry.stored_at.elapsed() > ttl {
            drop(entry);
            self.entries.remove(id);
            None
        } else {
            Some(entry.clone())
        }
    }

    /// Persist a completed response turn when `store = true`.
    pub fn insert(&self, id: &str, system: Option<String>, messages: Vec<ChatMessage>) {
        self.entries.insert(
            id.to_string(),
            StoredResponse {
                system,
                messages,
                stored_at: Instant::now(),
            },
        );
    }

    /// Remove expired responses.
    pub fn evict_expired(&self, ttl: Duration) {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| entry.stored_at.elapsed() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired {
            self.entries.remove(&key);
        }
    }

    /// Spawn a background eviction task tied to `ttl_secs`.
    pub fn spawn_eviction_task(
        self: &Arc<Self>,
        ttl_secs: Arc<std::sync::atomic::AtomicU64>,
        shutdown: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let store = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let ttl = Duration::from_secs(ttl_secs.load(std::sync::atomic::Ordering::Relaxed).max(1));
                        store.evict_expired(ttl);
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        })
    }
}

/// OpenAI `POST /v1/responses` request body.
#[derive(Debug, Deserialize)]
pub struct OpenAiResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default = "default_store_true")]
    store: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

fn default_store_true() -> bool {
    true
}

/// OpenAI Responses endpoint (`POST /v1/responses`).
pub async fn responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<OpenAiResponsesRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(rejection) => {
            return GatewayError::BadRequest(format!(
                "invalid Responses request body: {}",
                truncate(&rejection.body_text(), 500)
            ))
            .into_openai_response();
        }
    };
    match responses_inner(state, headers, payload).await {
        Ok(response) => response,
        Err(error) => error.into_openai_response(),
    }
}

async fn responses_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: OpenAiResponsesRequest,
) -> Result<Response> {
    let (new_system, new_messages) = normalize_input(payload.instructions, payload.input)?;
    let ttl = Duration::from_secs(state.router().session_ttl_secs());

    let (system, messages) = if let Some(prev_id) = payload
        .previous_response_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        let prior = state.responses().get(prev_id, ttl).ok_or_else(|| {
            GatewayError::NotFound(format!("previous response {prev_id:?} not found"))
        })?;
        let merged_system = new_system.or(prior.system);
        let mut merged_messages = prior.messages;
        merged_messages.extend(new_messages);
        (merged_system, merged_messages)
    } else {
        (new_system, new_messages)
    };

    if messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "input must contain at least one non-system message or item".into(),
        ));
    }

    let response_format = parse_responses_text_format(payload.text)?;
    let first_user = first_user_text(&messages);
    let context = prepare(
        &state,
        &headers,
        payload.model.as_deref(),
        system.as_deref(),
        &first_user,
    )?;
    let max_tokens = payload.max_output_tokens.or(payload.max_tokens);
    let tools = normalize_tools(payload.tools);
    let history_before_reply = messages.clone();
    let history_system = system.clone();
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
    let routed = execute(&state, request).await?;
    let backend = routed.backend_id.clone();
    let fallback = routed.fallback_used;
    let driver_response = routed.response;
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let should_store = payload.store;

    if payload.stream {
        let config = state.config();
        let keep_alive = config.server.keep_alive_interval_secs;
        let stream_timeout = config.server.stream_timeout_secs.max(1);
        let mut chunks = into_chunk_stream(driver_response);
        let model = context.model.clone();
        let request_id = context.request_id.clone();
        let stream_cancel = cancel.clone();
        let created_at = unix_now();
        let store = state.responses().clone();
        let resp_id_for_store = response_id.clone();

        let stream = async_stream::stream! {
            let _guard = CancelOnDrop(stream_cancel.clone());
            let msg_id = format!("msg_{}", Uuid::new_v4().simple());
            let created_payload = json!({
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": model,
                    "store": should_store,
                    "output": [],
                }
            });
            yield Ok::<Event, Infallible>(
                Event::default()
                    .event("response.created")
                    .data(created_payload.to_string()),
            );

            let deadline = tokio::time::Instant::now() + Duration::from_secs(stream_timeout);
            let mut timed_out = false;
            let mut text_started = false;
            let mut full_text = String::new();
            let mut tool_calls: BTreeMap<usize, ToolCall> = BTreeMap::new();
            let mut final_usage: Option<Usage> = None;

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
                            "code": "stream_timeout",
                            "message": format!("stream timed out after {stream_timeout} seconds"),
                        }
                    });
                    yield Ok(Event::default().event("error").data(error.to_string()));
                    break;
                }
                match item {
                    Some(Ok(chunk)) => {
                        let StreamChunk { delta, tool_calls: tc_deltas, finish_reason, usage, .. } = chunk;
                        if usage.is_some() {
                            final_usage = usage;
                        }
                        if !delta.is_empty() {
                            if !text_started {
                                text_started = true;
                                let item_added = json!({
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {
                                        "id": msg_id,
                                        "type": "message",
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": [],
                                    }
                                });
                                yield Ok(Event::default()
                                    .event("response.output_item.added")
                                    .data(item_added.to_string()));
                                let part_added = json!({
                                    "type": "response.content_part.added",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": "", "annotations": []},
                                });
                                yield Ok(Event::default()
                                    .event("response.content_part.added")
                                    .data(part_added.to_string()));
                            }
                            full_text.push_str(&delta);
                            let delta_payload = json!({
                                "type": "response.output_text.delta",
                                "item_id": msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": delta,
                            });
                            yield Ok(Event::default()
                                .event("response.output_text.delta")
                                .data(delta_payload.to_string()));
                        }
                        for tc in tc_deltas {
                            let entry = tool_calls.entry(tc.index).or_insert_with(|| ToolCall {
                                id: format!("call_{}", Uuid::new_v4().simple()),
                                name: String::new(),
                                arguments: String::new(),
                            });
                            if let Some(id) = tc.id {
                                entry.id = id;
                            }
                            if let Some(name) = tc.name {
                                entry.name.push_str(&name);
                            }
                            if let Some(args) = tc.arguments_delta {
                                entry.arguments.push_str(&args);
                                let delta_payload = json!({
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": entry.id,
                                    "output_index": tc.index + usize::from(text_started),
                                    "delta": args,
                                });
                                yield Ok(Event::default()
                                    .event("response.function_call_arguments.delta")
                                    .data(delta_payload.to_string()));
                            }
                        }
                        if finish_reason.is_some() {
                            if text_started {
                                let text_done = json!({
                                    "type": "response.output_text.done",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "text": full_text,
                                });
                                yield Ok(Event::default()
                                    .event("response.output_text.done")
                                    .data(text_done.to_string()));
                            }
                            let collected_calls: Vec<ToolCall> = tool_calls.into_values().collect();
                            if should_store {
                                let mut full_history = history_before_reply;
                                full_history.push(ChatMessage {
                                    role: "assistant".into(),
                                    content: MessageContent::Text(full_text.clone()),
                                    tool_calls: collected_calls.clone(),
                                    tool_call_id: None,
                                    name: None,
                                });
                                store.insert(&resp_id_for_store, history_system, full_history);
                            }
                            let completed_body = build_response_object(
                                &response_id,
                                created_at,
                                &model,
                                &msg_id,
                                should_store,
                                &full_text,
                                &collected_calls,
                                final_usage.unwrap_or_default(),
                            );
                            let completed = json!({
                                "type": "response.completed",
                                "response": completed_body,
                            });
                            yield Ok(Event::default()
                                .event("response.completed")
                                .data(completed.to_string()));
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        let payload = json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "code": "stream_error",
                                "message": error.to_string(),
                            }
                        });
                        yield Ok(Event::default().event("error").data(payload.to_string()));
                        break;
                    }
                    None => {
                        if text_started {
                            let text_done = json!({
                                "type": "response.output_text.done",
                                "item_id": msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": full_text,
                            });
                            yield Ok(Event::default()
                                .event("response.output_text.done")
                                .data(text_done.to_string()));
                        }
                        let collected_calls: Vec<ToolCall> = tool_calls.into_values().collect();
                        if should_store {
                            let mut full_history = history_before_reply;
                            full_history.push(ChatMessage {
                                role: "assistant".into(),
                                content: MessageContent::Text(full_text.clone()),
                                tool_calls: collected_calls.clone(),
                                tool_call_id: None,
                                name: None,
                            });
                            store.insert(&resp_id_for_store, history_system, full_history);
                        }
                        let completed_body = build_response_object(
                            &response_id,
                            created_at,
                            &model,
                            &msg_id,
                            should_store,
                            &full_text,
                            &collected_calls,
                            final_usage.unwrap_or_default(),
                        );
                        let completed = json!({
                            "type": "response.completed",
                            "response": completed_body,
                        });
                        yield Ok(Event::default()
                            .event("response.completed")
                            .data(completed.to_string()));
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
    if should_store {
        let mut full_history = history_before_reply;
        full_history.push(ChatMessage {
            role: "assistant".into(),
            content: MessageContent::Text(completion.content.clone()),
            tool_calls: completion.tool_calls.clone(),
            tool_call_id: None,
            name: None,
        });
        state
            .responses()
            .insert(&response_id, history_system, full_history);
    }
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());
    let body = build_response_object(
        &response_id,
        unix_now(),
        &context.model,
        &msg_id,
        should_store,
        &completion.content,
        &completion.tool_calls,
        completion.usage,
    );
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

#[allow(clippy::too_many_arguments)]
fn build_response_object(
    response_id: &str,
    created_at: u64,
    model: &str,
    msg_id: &str,
    store: bool,
    content: &str,
    tool_calls: &[ToolCall],
    usage: Usage,
) -> Value {
    let mut output = Vec::new();
    if !content.is_empty() || tool_calls.is_empty() {
        output.push(json!({
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": [],
            }],
        }));
    }
    for call in tool_calls {
        output.push(json!({
            "id": format!("fc_{}", Uuid::new_v4().simple()),
            "type": "function_call",
            "status": "completed",
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        }));
    }
    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "store": store,
        "output": output,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    })
}

fn parse_responses_text_format(text: Option<Value>) -> Result<Option<ResponseFormat>> {
    let Some(text_obj) = text else {
        return Ok(None);
    };
    let Some(format) = text_obj.get("format").cloned() else {
        return Ok(None);
    };
    parse_openai_response_format(Some(format))
}

fn normalize_input(
    instructions: Option<String>,
    input: Option<Value>,
) -> Result<(Option<String>, Vec<ChatMessage>)> {
    let mut system_parts = Vec::new();
    if let Some(instructions) = instructions
        && !instructions.trim().is_empty()
    {
        system_parts.push(instructions);
    }
    let mut messages = Vec::new();

    match input {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => {
            if !text.trim().is_empty() {
                messages.push(ChatMessage::text("user", text));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                let item_type = item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message");
                match item_type {
                    "function_call" => {
                        let id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                            .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                            .to_string();
                        let arguments = match item.get("arguments") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => "{}".to_string(),
                        };
                        messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: MessageContent::Text(String::new()),
                            tool_calls: vec![ToolCall {
                                id,
                                name,
                                arguments,
                            }],
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    "function_call_output" => {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .map(ToString::to_string);
                        let output = item
                            .get("output")
                            .and_then(content_to_text)
                            .unwrap_or_default();
                        messages.push(ChatMessage {
                            role: "tool".into(),
                            content: MessageContent::Text(output),
                            tool_calls: Vec::new(),
                            tool_call_id: call_id,
                            name: None,
                        });
                    }
                    "input_image" => {
                        let parsed = parse_json_content(&item);
                        if !parsed.is_empty() {
                            messages.push(ChatMessage {
                                role: "user".into(),
                                content: parsed,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                name: None,
                            });
                        }
                    }
                    _ => {
                        let role = item
                            .get("role")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|r| !r.is_empty())
                            .unwrap_or("user");
                        let content = if let Some(content_val) = item.get("content") {
                            parse_json_content(content_val)
                        } else if let Some(text) = item.get("text").and_then(Value::as_str) {
                            MessageContent::Parts(vec![ContentPart::Text {
                                text: text.to_string(),
                                cache_control: None,
                            }])
                        } else {
                            MessageContent::default()
                        };
                        if role == "system" || role == "developer" {
                            let text = content.as_plain_text();
                            if !text.trim().is_empty() {
                                system_parts.push(text);
                            }
                        } else if !content.is_empty() {
                            messages.push(ChatMessage {
                                role: role.to_string(),
                                content,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                name: None,
                            });
                        }
                    }
                }
            }
        }
        Some(_) => {
            return Err(GatewayError::BadRequest(
                "input must be a string or an array of response items".into(),
            ));
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, messages))
}
