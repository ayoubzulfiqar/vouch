//! Connection-pooled HTTP/SSE proxy driver.
//!
//! This is the preferred transport for local daemons that already expose an
//! OpenAI-compatible API (`hermes proxy`, `opencode serve`). The driver
//! normalizes both OpenAI and Anthropic SSE deltas into the gateway's
//! [`StreamChunk`] shape.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::{
    Client,
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    config::{BackendConfig, BackendKind, DefaultsConfig},
    driver::{
        ChunkStream, Completion, Driver, DriverResponse, EmbeddingRequest, EmbeddingResult,
        ExecutionRequest, PermitGuard, SseDecoder, StreamChunk, ToolCallDelta, Usage,
        extract_content, extract_delta_full, extract_thinking, extract_tool_call_deltas,
        extract_tool_calls, extract_upstream_error, extract_usage,
        map_upstream_status_with_headers, truncate,
    },
    error::{GatewayError, Result},
    telemetry,
};

/// HTTP/SSE passthrough driver.
#[derive(Debug)]
pub struct HttpProxyDriver {
    config: BackendConfig,
    client: Client,
    semaphore: Arc<Semaphore>,
    queue_timeout: Duration,
    chat_url: String,
    health_url: Option<String>,
    api_key: Option<String>,
}

impl HttpProxyDriver {
    /// Build a driver from backend configuration.
    pub fn new(config: BackendConfig, defaults: &DefaultsConfig) -> Result<Self> {
        let base = config.base_url.as_deref().ok_or_else(|| {
            GatewayError::Config(format!("backend {} has no base_url", config.id))
        })?;
        let base = base.trim_end_matches('/');
        let upstream_path = config
            .upstream_path
            .as_deref()
            .unwrap_or("/v1/chat/completions");
        let chat_url = format!("{base}/{}", upstream_path.trim_start_matches('/'));
        let health_url = config
            .health_path
            .as_deref()
            .map(|path| format!("{base}/{}", path.trim_start_matches('/')));

        let client = Client::builder()
            .pool_max_idle_per_host(config.max_concurrency.saturating_mul(2).max(4))
            .tcp_keepalive(Duration::from_secs(30))
            .timeout(Duration::from_secs(config.timeout_secs.max(1)))
            .build()
            .map_err(|error| {
                GatewayError::Internal(format!(
                    "cannot build HTTP client for {}: {error}",
                    config.id
                ))
            })?;

        let api_key = config
            .api_key_env
            .as_ref()
            .and_then(|name| std::env::var(name).ok())
            .filter(|value| !value.is_empty());

        let queue_timeout = Duration::from_millis(
            config
                .queue_timeout_ms
                .unwrap_or(defaults.queue_timeout_ms)
                .max(1),
        );

        Ok(Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrency.max(1))),
            config,
            client,
            queue_timeout,
            chat_url,
            health_url,
            api_key,
        })
    }

    fn headers(
        &self,
        stream: bool,
        trace: Option<&crate::telemetry::TraceContext>,
    ) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        } else {
            headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        }
        if let Some(trace) = trace {
            if let Ok(val) = HeaderValue::from_str(&trace.traceparent()) {
                headers.insert(HeaderName::from_static("traceparent"), val);
            }
            if let Some(ts) = trace.tracestate.as_deref()
                && let Ok(val) = HeaderValue::from_str(ts)
            {
                headers.insert(HeaderName::from_static("tracestate"), val);
            }
        }
        if let Some(key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| {
                GatewayError::Config(format!(
                    "backend {}: api key contains invalid header bytes",
                    self.config.id
                ))
            })?;
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        for (name, value) in &self.config.extra_headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                GatewayError::Config(format!(
                    "backend {}: invalid extra header name {name:?}: {error}",
                    self.config.id
                ))
            })?;
            let value = HeaderValue::from_str(value).map_err(|error| {
                GatewayError::Config(format!(
                    "backend {}: invalid extra header value for {name}: {error}",
                    self.config.id
                ))
            })?;
            headers.insert(name, value);
        }
        Ok(headers)
    }

    fn body(&self, request: &ExecutionRequest) -> Value {
        let mut body = json!({
            "model": self.config.model.clone().unwrap_or_else(|| request.model.clone()),
            "messages": request.openai_messages(),
            "stream": request.stream,
        });
        if let Some(temperature) = request.temperature
            && let Some(object) = body.as_object_mut()
        {
            object.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens
            && let Some(object) = body.as_object_mut()
        {
            object.insert("max_tokens".into(), json!(max_tokens));
        }
        if !request.tools.is_empty()
            && let Some(object) = body.as_object_mut()
        {
            object.insert("tools".into(), json!(request.tools));
        }
        if let Some(tool_choice) = &request.tool_choice
            && let Some(object) = body.as_object_mut()
        {
            object.insert("tool_choice".into(), tool_choice.clone());
        }
        if let Some(format) = &request.response_format
            && let Some(object) = body.as_object_mut()
        {
            object.insert("response_format".into(), format.to_openai_json());
        }
        if let Some(object) = body.as_object_mut() {
            for (key, value) in &request.extra {
                if !matches!(
                    key.as_str(),
                    "model" | "messages" | "stream" | "tools" | "tool_choice" | "response_format"
                ) {
                    object.insert(key.clone(), value.clone());
                }
            }
        }
        body
    }

    async fn complete(&self, request: &ExecutionRequest) -> Result<Completion> {
        let response = self
            .client
            .post(&self.chat_url)
            .headers(self.headers(false, request.trace_context.as_ref())?)
            .json(&self.body(request))
            .send()
            .await?;
        let status = response.status();
        let resp_headers = response.headers().clone();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(map_upstream_status_with_headers(
                status.as_u16(),
                Some(&resp_headers),
                &text,
            ));
        }
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            GatewayError::Upstream(format!(
                "backend {} returned invalid JSON: {error}",
                self.config.id
            ))
        })?;
        if let Some(error) = extract_upstream_error(&value) {
            return Err(GatewayError::Upstream(format!(
                "backend {} reported an error: {}",
                self.config.id,
                truncate(&telemetry::redact_text(&error), 500)
            )));
        }
        let tool_calls = extract_tool_calls(&value);
        let thinking = extract_thinking(&value);
        let content = match extract_content(&value) {
            Some(content) => content,
            None if !tool_calls.is_empty() || !thinking.is_empty() => String::new(),
            None => {
                return Err(GatewayError::Upstream(format!(
                    "backend {} response did not contain assistant content",
                    self.config.id
                )));
            }
        };
        let default_finish = if tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let finish_reason = value
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or(default_finish)
            .to_string();
        let usage = extract_usage(&value).unwrap_or_else(|| {
            Usage::estimate_for_model(&request.model, &request.rendered_prompt(), &content)
        });
        Ok(Completion {
            content,
            finish_reason,
            usage,
            raw: Some(value),
            tool_calls,
            thinking,
        })
    }

    async fn stream(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let response = self
            .client
            .post(&self.chat_url)
            .headers(self.headers(true, request.trace_context.as_ref())?)
            .json(&self.body(&request))
            .send()
            .await?;
        let status = response.status();
        let resp_headers = response.headers().clone();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(map_upstream_status_with_headers(
                status.as_u16(),
                Some(&resp_headers),
                &text,
            ));
        }
        let is_sse = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            // Some daemons ignore `stream: true`. Fall back to a complete
            // response delivered as streamed chunks.
            let text = response.text().await?;
            let value: Value = serde_json::from_str(&text).map_err(|error| {
                GatewayError::Upstream(format!(
                    "backend {} returned invalid JSON: {error}",
                    self.config.id
                ))
            })?;
            let tool_calls = extract_tool_calls(&value);
            let thinking = extract_thinking(&value);
            let content = match extract_content(&value) {
                Some(content) => content,
                None if !tool_calls.is_empty() || !thinking.is_empty() => String::new(),
                None => {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} response did not contain assistant content",
                        self.config.id
                    )));
                }
            };
            let usage = extract_usage(&value);
            let mut items = Vec::new();
            for block in thinking {
                match block {
                    crate::driver::ThinkingBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        let mut chunk = StreamChunk::thinking(thinking);
                        chunk.signature_delta = signature;
                        items.push(Ok(chunk));
                    }
                    crate::driver::ThinkingBlock::RedactedThinking { data } => {
                        let mut chunk = StreamChunk::delta("");
                        chunk.redacted_thinking = Some(data);
                        items.push(Ok(chunk));
                    }
                }
            }
            if !content.is_empty() {
                items.push(Ok(StreamChunk::delta(content)));
            }
            if !tool_calls.is_empty() {
                let deltas = tool_calls
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
                items.push(Ok(StreamChunk::finish("tool_calls", usage)));
            } else {
                items.push(Ok(StreamChunk::finish("stop", usage)));
            }
            let stream: ChunkStream = Box::pin(futures::stream::iter(items));
            return Ok(DriverResponse::Stream(stream));
        }

        let backend = self.config.id.clone();
        let cancel = request.cancel.clone();
        let model = request.model.clone();
        let prompt = request.rendered_prompt();
        let (sender, receiver) = mpsc::channel::<Result<StreamChunk>>(32);
        let mut bytes = response.bytes_stream();

        tokio::spawn(async move {
            let _permit = permit;
            let mut decoder = SseDecoder::new();
            let mut usage = None;
            let mut finished = false;
            let mut saw_tool_calls = false;
            let mut produced = String::new();

            loop {
                let chunk = tokio::select! {
                    _ = cancel.cancelled() => return,
                    item = bytes.next() => item,
                };
                let Some(chunk) = chunk else {
                    break;
                };
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let _ = sender
                            .send(Err(GatewayError::Upstream(format!(
                                "backend {backend} stream error: {error}"
                            ))))
                            .await;
                        return;
                    }
                };
                for event in decoder.push(&chunk) {
                    let data = event.data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        finished = true;
                        break;
                    }
                    let value: Value = match serde_json::from_str(data) {
                        Ok(value) => value,
                        Err(_) => continue,
                    };
                    if let Some(error) = extract_upstream_error(&value) {
                        let _ = sender
                            .send(Err(GatewayError::Upstream(format!(
                                "backend {backend} reported an error: {}",
                                truncate(&telemetry::redact_text(&error), 500)
                            ))))
                            .await;
                        return;
                    }
                    let parts = extract_delta_full(&value);
                    let tool_deltas = extract_tool_call_deltas(&value);
                    if parts.usage.is_some() {
                        usage = parts.usage;
                    }
                    if let Some(redacted) = parts.redacted_thinking {
                        let mut c = StreamChunk::delta("");
                        c.redacted_thinking = Some(redacted);
                        if sender.send(Ok(c)).await.is_err() {
                            return;
                        }
                    }
                    if parts.thinking.is_some() || parts.signature.is_some() {
                        let mut c = StreamChunk::delta("");
                        c.thinking_delta = parts.thinking;
                        c.signature_delta = parts.signature;
                        if sender.send(Ok(c)).await.is_err() {
                            return;
                        }
                    }
                    if let Some(delta) = parts.content
                        && !delta.is_empty()
                    {
                        produced.push_str(&delta);
                        if sender.send(Ok(StreamChunk::delta(delta))).await.is_err() {
                            return;
                        }
                    }
                    if !tool_deltas.is_empty() {
                        saw_tool_calls = true;
                        if sender
                            .send(Ok(StreamChunk::tool_calls(tool_deltas)))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    if let Some(reason) = parts.finish_reason {
                        let _ = sender.send(Ok(StreamChunk::finish(reason, usage))).await;
                        finished = true;
                        break;
                    }
                }
                if finished {
                    break;
                }
            }

            if !finished {
                let usage =
                    usage.or_else(|| Some(Usage::estimate_for_model(&model, &prompt, &produced)));
                let reason = if saw_tool_calls { "tool_calls" } else { "stop" };
                let _ = sender.send(Ok(StreamChunk::finish(reason, usage))).await;
            }
        });

        let stream: ChunkStream = Box::pin(ReceiverStream::new(receiver));
        Ok(DriverResponse::Stream(stream))
    }
}

#[async_trait]
impl Driver for HttpProxyDriver {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Http
    }

    fn max_concurrency(&self) -> usize {
        self.config.max_concurrency
    }

    async fn execute(&self, request: ExecutionRequest) -> Result<DriverResponse> {
        let permit = PermitGuard::acquire(
            self.semaphore.clone(),
            &self.config.id,
            self.queue_timeout,
            &request.cancel,
        )
        .await?;
        if request.stream {
            self.stream(request, permit).await
        } else {
            let completion = self.complete(&request).await?;
            permit.release();
            Ok(DriverResponse::Complete(completion))
        }
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResult> {
        let permit = PermitGuard::acquire(
            self.semaphore.clone(),
            &self.config.id,
            self.queue_timeout,
            &request.cancel,
        )
        .await?;
        let base = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("")
            .trim_end_matches('/');
        let upstream_path = self.config.upstream_path.as_deref().unwrap_or("");
        let is_ollama = upstream_path.starts_with("/api/");
        let embed_url =
            if upstream_path.ends_with("/embeddings") || upstream_path.ends_with("/embed") {
                format!("{base}/{}", upstream_path.trim_start_matches('/'))
            } else if is_ollama {
                format!("{base}/api/embed")
            } else {
                format!("{base}/v1/embeddings")
            };

        let upstream_model = self
            .config
            .model
            .clone()
            .unwrap_or_else(|| request.model.clone());
        let mut payload = json!({
            "model": upstream_model,
            "input": request.inputs,
        });
        if !is_ollama
            && let Some(dim) = request.dimensions
            && let Some(obj) = payload.as_object_mut()
        {
            obj.insert("dimensions".into(), json!(dim));
        }

        let response = self
            .client
            .post(&embed_url)
            .headers(self.headers(false, request.trace_context.as_ref())?)
            .json(&payload)
            .send()
            .await?;
        let status = response.status();
        let resp_headers = response.headers().clone();
        let text = response.text().await?;
        permit.release();

        if !status.is_success() {
            return Err(map_upstream_status_with_headers(
                status.as_u16(),
                Some(&resp_headers),
                &text,
            ));
        }
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            GatewayError::Upstream(format!(
                "backend {} returned invalid embeddings JSON: {error}",
                self.config.id
            ))
        })?;
        if let Some(error) = extract_upstream_error(&value) {
            return Err(GatewayError::Upstream(format!(
                "backend {} reported an embeddings error: {}",
                self.config.id,
                truncate(&telemetry::redact_text(&error), 500)
            )));
        }

        let embeddings = parse_embeddings_json(&self.config.id, &value)?;
        let prompt_tokens = value
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .or_else(|| value.get("prompt_eval_count").and_then(Value::as_u64))
            .unwrap_or(0);
        Ok(EmbeddingResult {
            embeddings,
            usage: Usage::new(prompt_tokens, 0),
            model: upstream_model,
        })
    }

    async fn probe(&self) -> std::result::Result<(), String> {
        let url = self
            .health_url
            .clone()
            .unwrap_or_else(|| self.chat_url.clone());
        let response = self
            .client
            .get(url)
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|error| format!("probe failed: {error}"))?;
        if response.status().is_success() || response.status().as_u16() == 405 {
            Ok(())
        } else {
            Err(format!("probe returned HTTP {}", response.status()))
        }
    }
}

/// Helper used by tests and diagnostics: build an OpenAI request body.
pub fn build_openai_body(model: &str, messages: Vec<Value>, stream: bool) -> Map<String, Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(stream));
    body
}

fn parse_embeddings_json(backend_id: &str, value: &Value) -> Result<Vec<Vec<f32>>> {
    if let Some(data) = value.get("data").and_then(Value::as_array) {
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let vec_vals = item
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GatewayError::Upstream(format!(
                        "backend {backend_id} embeddings data item missing vector"
                    ))
                })?;
            let vec: Vec<f32> = vec_vals
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            out.push(vec);
        }
        return Ok(out);
    }
    if let Some(arrays) = value.get("embeddings").and_then(Value::as_array) {
        let mut out = Vec::with_capacity(arrays.len());
        for arr in arrays {
            let vec_vals = arr.as_array().ok_or_else(|| {
                GatewayError::Upstream(format!(
                    "backend {backend_id} ollama embeddings entry is not an array"
                ))
            })?;
            out.push(
                vec_vals
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect(),
            );
        }
        return Ok(out);
    }
    Err(GatewayError::Upstream(format!(
        "backend {backend_id} returned unrecognized embeddings payload"
    )))
}
