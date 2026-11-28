//! Operational endpoints: liveness, readiness, and Prometheus metrics.
//!
//! These endpoints are intentionally unauthenticated. They expose only
//! backend identifiers and circuit state, never credentials or prompts.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{VERSION, state::AppState};

/// Liveness probe.
pub async fn healthz(State(state): State<Arc<AppState>>) -> Response {
    let config = state.config();
    Json(json!({
        "status": "ok",
        "version": VERSION,
        "uptime_secs": state.uptime_secs(),
        "models": state.router().models().len(),
        "backends": state.router().backends().len(),
        "tokens_enforced": state.tokens().enabled(),
        "workspace_enabled": config.workspace.enabled,
    }))
    .into_response()
}

/// Readiness probe based on circuit state.
pub async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    let report = state.router().readiness();
    let status = if report.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
}

/// Prometheus exposition.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.metrics().render(),
    )
        .into_response()
}

/// Cumulative per-token and per-backend usage and latency analytics (`GET /v1/usage`).
pub async fn usage(State(state): State<Arc<AppState>>, headers: axum::http::HeaderMap) -> Response {
    if let Err(error) = state.tokens().authenticate(&headers) {
        return error.into_openai_response();
    }
    let budgets = state.tokens().budget_statuses();
    let report = state.usage().report(state.uptime_secs(), &budgets);
    Json(report).into_response()
}

/// Recent completed request traces ring buffer (`GET /v1/traces`).
pub async fn traces(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(error) = state.tokens().authenticate(&headers) {
        return error.into_openai_response();
    }
    let traces = state.traces().recent();
    Json(json!({ "traces": traces })).into_response()
}
