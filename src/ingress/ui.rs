//! SPDX-License-Identifier: GPL-3.0-or-later
//! Optional embedded read-only operational dashboard (`GET /ui`).
//!
//! Disabled by default (`server.enable_ui = false`) so the daemon adds zero
//! browser-facing attack surface unless explicitly opted in.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
};

use crate::state::AppState;

const DASHBOARD_HTML: &str = include_str!("ui.html");

/// Serve the read-only operational HTML dashboard when `server.enable_ui = true`.
pub async fn dashboard(State(state): State<Arc<AppState>>) -> Response {
    if !state.config().server.enable_ui {
        return StatusCode::NOT_FOUND.into_response();
    }
    let mut response = Html(DASHBOARD_HTML).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}
