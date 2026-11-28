//! SPDX-License-Identifier: GPL-3.0-or-later
//! Shared cross-agent session ledger inspection, checkpoint, and teardown endpoints
//! (`/v1/sessions`, `/v1/sessions/{session_id}`, `/v1/sessions/{session_id}/checkpoint`).

use std::{sync::Arc, time::Duration};

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;

use crate::{
    error::{GatewayError, Result},
    ingress::header_text,
    memory::{ledger::format_trace_session_summary, resolve_workspace_root_header},
    state::AppState,
};

#[derive(Debug, Deserialize)]
pub struct CheckpointRequest {
    pub note: String,
    #[serde(default)]
    pub files: Vec<String>,
}

/// List all live shared session ledgers (`GET /v1/sessions`).
pub async fn list(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Result<Response> {
    let _principal = state.tokens().authenticate(&headers)?;
    let ttl = Duration::from_secs(state.config().defaults.session_ttl_secs.max(1));
    let sessions = state.memory().ledger().list(ttl);
    Ok((StatusCode::OK, Json(json!({ "sessions": sessions }))).into_response())
}

/// Inspect one canonical shared session ledger (`GET /v1/sessions/{session_id}`).
pub async fn get(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response> {
    let _principal = state.tokens().authenticate(&headers)?;
    let trimmed = session_id.trim();
    if trimmed.is_empty() {
        return Err(GatewayError::BadRequest(
            "session_id must not be empty".into(),
        ));
    }
    let ttl = Duration::from_secs(state.config().defaults.session_ttl_secs.max(1));
    let entry = state
        .memory()
        .ledger()
        .get(trimmed, ttl)
        .ok_or_else(|| GatewayError::NotFound(format!("session {trimmed:?} not found")))?;
    Ok((StatusCode::OK, Json(entry)).into_response())
}

/// Append a deterministic checkpoint note and optionally sync to `[trace]`
/// (`POST /v1/sessions/{session_id}/checkpoint`).
pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<CheckpointRequest>,
) -> Result<Response> {
    let _principal = state.tokens().authenticate(&headers)?;
    let trimmed_id = session_id.trim();
    if trimmed_id.is_empty() {
        return Err(GatewayError::BadRequest(
            "session_id must not be empty".into(),
        ));
    }
    let trimmed_note = payload.note.trim();
    if trimmed_note.is_empty() {
        return Err(GatewayError::BadRequest(
            "checkpoint note must not be empty".into(),
        ));
    }

    let cfg = state.config();
    let ws_override = resolve_workspace_root_header(
        &cfg.trace,
        &cfg.workspace,
        header_text(&headers, "x-workspace-root"),
    )?;
    let effective_ws = ws_override
        .clone()
        .unwrap_or_else(|| state.memory().trace().default_workspace_root());

    let entry = state.memory().ledger().add_checkpoint(
        trimmed_id,
        trimmed_note,
        &payload.files,
        Some(&effective_ws),
    );

    let synced_to_trace = if cfg.shared_memory.sync_to_trace && cfg.trace.enabled {
        let summary = format_trace_session_summary(&entry);
        state
            .memory()
            .trace()
            .record_session_checkpoint_now(ws_override.as_deref(), &summary, &entry.touched_files)
            .await
            .unwrap_or(false)
    } else {
        false
    };

    Ok((
        StatusCode::OK,
        Json(json!({
            "session": entry,
            "synced_to_trace": synced_to_trace,
        })),
    )
        .into_response())
}

/// Delete a shared session ledger, router affinity, and any ephemeral git worktree
/// (`DELETE /v1/sessions/{session_id}`).
pub async fn delete(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response> {
    let _principal = state.tokens().authenticate(&headers)?;
    let trimmed = session_id.trim();
    if trimmed.is_empty() {
        return Err(GatewayError::BadRequest(
            "session_id must not be empty".into(),
        ));
    }
    let ledger_removed = state.memory().ledger().remove(trimmed);
    let route_removed = state.router().sessions().remove_session(trimmed);
    let worktree_removed = state.workspaces().remove_session(trimmed).await;
    if !ledger_removed && !route_removed && !worktree_removed {
        return Err(GatewayError::NotFound(format!(
            "session {trimmed:?} not found"
        )));
    }
    Ok((
        StatusCode::OK,
        Json(json!({
            "deleted": true,
            "session_id": trimmed,
            "worktree_removed": worktree_removed,
        })),
    )
        .into_response())
}
