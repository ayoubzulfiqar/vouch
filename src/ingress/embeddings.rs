//! SPDX-License-Identifier: GPL-3.0-or-later
//! OpenAI-compatible embeddings ingress (`POST /v1/embeddings`).

use std::{sync::Arc, time::Instant};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    driver::{ChatMessage, EmbeddingRequest, Usage, truncate},
    error::{GatewayError, Result},
    ingress::{
        insert_routing_headers, prepare_routed, record_completion_outcome, record_error_outcome,
        start_ingress_span,
    },
    state::AppState,
};

/// OpenAI `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingsRequest {
    #[serde(default)]
    model: Option<String>,
    input: Value,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<u32>,
}

/// Handle `POST /v1/embeddings`.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: std::result::Result<Json<OpenAiEmbeddingsRequest>, JsonRejection>,
) -> Response {
    let Json(payload) = match payload {
        Ok(p) => p,
        Err(rejection) => {
            return GatewayError::BadRequest(format!(
                "invalid embeddings request body: {}",
                truncate(&rejection.body_text(), 500)
            ))
            .into_openai_response();
        }
    };
    match embeddings_inner(state, headers, payload).await {
        Ok(response) => response,
        Err(error) => error.into_openai_response(),
    }
}

async fn embeddings_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: OpenAiEmbeddingsRequest,
) -> Result<Response> {
    let mut inputs = parse_inputs(payload.input)?;
    let encoding_format = match payload.encoding_format.as_deref() {
        None => "float",
        Some(f) if f.eq_ignore_ascii_case("float") => "float",
        Some(f) if f.eq_ignore_ascii_case("base64") => "base64",
        Some(other) => {
            return Err(GatewayError::BadRequest(format!(
                "unsupported encoding_format {other:?}; expected \"float\" or \"base64\""
            )));
        }
    };

    let messages: Vec<ChatMessage> = inputs
        .iter()
        .map(|s| ChatMessage::text("user", s.clone()))
        .collect();
    let context = prepare_routed(
        &state,
        &headers,
        payload.model.as_deref(),
        None,
        &messages,
        &[],
    )?;
    let started = Instant::now();
    let ingress_span = start_ingress_span(&state, &context, "/v1/embeddings");

    if let Err(error) = state.egress_guard().inspect_texts(&mut inputs) {
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

    let request = EmbeddingRequest {
        request_id: context.request_id.clone(),
        model: context.model.clone(),
        inputs: inputs.clone(),
        encoding_format: Some(encoding_format.to_string()),
        dimensions: payload.dimensions,
        cancel: CancellationToken::new(),
        trace_context: Some(context.trace_context.clone()),
    };

    let timeout = std::time::Duration::from_secs(state.config().server.request_timeout_secs.max(1));
    let allowed_backends = context.principal.allowed_backends();
    let routed = match tokio::time::timeout(
        timeout,
        state
            .router()
            .execute_embeddings(&context.model, allowed_backends.as_deref(), request),
    )
    .await
    {
        Ok(Ok(routed)) => routed,
        Ok(Err(error)) => {
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
        Err(_) => {
            let error = GatewayError::Timeout(timeout);
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

    let effective_usage = if routed.result.usage.total_tokens == 0 {
        Usage::new(context.prompt_tokens, 0)
    } else {
        routed.result.usage
    };
    context.set_backend_meta(
        routed.backend_id.clone(),
        routed.profile_id.clone(),
        routed.fallback_used,
        false,
        "disabled",
        started.elapsed().as_millis() as u64,
        None,
    );
    record_completion_outcome(
        &state,
        &context,
        &routed.backend_id,
        routed.fallback_used,
        effective_usage,
        started.elapsed(),
        Some(ingress_span),
    );

    let data: Vec<Value> = routed
        .result
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(index, vec)| {
            let embedding_val = if encoding_format == "base64" {
                let mut raw_bytes = Vec::with_capacity(vec.len() * 4);
                for f in vec {
                    raw_bytes.extend_from_slice(&f.to_le_bytes());
                }
                Value::String(base64::engine::general_purpose::STANDARD.encode(&raw_bytes))
            } else {
                json!(vec)
            };
            json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding_val,
            })
        })
        .collect();

    let body = json!({
        "object": "list",
        "data": data,
        "model": context.model,
        "usage": {
            "prompt_tokens": effective_usage.prompt_tokens,
            "total_tokens": effective_usage.total_tokens,
        }
    });
    let mut response = (StatusCode::OK, Json(body)).into_response();
    insert_routing_headers(
        &mut response,
        &routed.backend_id,
        routed.fallback_used,
        &context.request_id,
        None,
    );
    Ok(response)
}

fn parse_inputs(input: Value) -> Result<Vec<String>> {
    match input {
        Value::String(s) => {
            if s.is_empty() {
                return Err(GatewayError::BadRequest(
                    "input string must not be empty".into(),
                ));
            }
            Ok(vec![s])
        }
        Value::Array(items) => {
            if items.is_empty() {
                return Err(GatewayError::BadRequest(
                    "input array must not be empty".into(),
                ));
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(s) => out.push(s),
                    _ => {
                        return Err(GatewayError::BadRequest(
                            "input array items must be strings".into(),
                        ));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(GatewayError::BadRequest(
            "input must be a string or an array of strings".into(),
        )),
    }
}
