//! SPDX-License-Identifier: GPL-3.0-or-later
//! Runtime administrative control endpoints (`POST /admin/reload`,
//! `POST /admin/circuits/{backend}/reset`, `DELETE /admin/cache`).

use std::{convert::Infallible, net::SocketAddr, sync::Arc};

use axum::{
    Json,
    extract::{ConnectInfo, FromRequestParts, Path, State},
    http::{HeaderMap, request::Parts},
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{
    error::{GatewayError, Result},
    state::AppState,
};

/// Optional peer socket address extracted from `ConnectInfo<SocketAddr>` when running over TCP.
pub struct OptionalPeer(pub Option<SocketAddr>);

impl<S: Send + Sync> FromRequestParts<S> for OptionalPeer {
    type Rejection = Infallible;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> std::result::Result<Self, Self::Rejection> {
        Ok(Self(
            parts
                .extensions
                .get::<ConnectInfo<SocketAddr>>()
                .map(|c| c.0),
        ))
    }
}

fn authorize_admin(state: &AppState, headers: &HeaderMap, peer: Option<SocketAddr>) -> Result<()> {
    if !state.config().server.enable_admin {
        return Err(GatewayError::NotFound(
            "admin endpoints are disabled (enable server.enable_admin = true)".into(),
        ));
    }
    if state.tokens().enabled() {
        let principal = state.tokens().authenticate(headers)?;
        if !principal.is_admin() {
            return Err(GatewayError::Forbidden(format!(
                "token {:?} lacks admin = true permission",
                principal.name
            )));
        }
        return Ok(());
    }
    if let Some(addr) = peer
        && !addr.ip().is_loopback()
    {
        return Err(GatewayError::Forbidden(
            "unauthenticated admin requests require a loopback source address".into(),
        ));
    }
    Ok(())
}

/// `POST /admin/reload` — atomically reload TOML configuration without dropping in-flight streams.
pub async fn reload(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OptionalPeer(peer): OptionalPeer,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers, peer) {
        return error.into_openai_response();
    }
    match state.reload().await {
        Ok(cfg) => Json(json!({
            "status": "reloaded",
            "models": cfg.routes.len(),
            "backends": cfg.backends.len(),
        }))
        .into_response(),
        Err(error) => error.into_openai_response(),
    }
}

/// `POST /admin/circuits/{backend}/reset` — manually close/reset a backend circuit breaker.
pub async fn reset_circuit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OptionalPeer(peer): OptionalPeer,
    Path(backend): Path<String>,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers, peer) {
        return error.into_openai_response();
    }
    if !state.router().reset_circuit(&backend) {
        return GatewayError::NotFound(format!("unknown backend {backend:?}"))
            .into_openai_response();
    }
    Json(json!({
        "status": "reset",
        "backend": backend,
    }))
    .into_response()
}

/// `DELETE /admin/cache` — purge every entry in the deterministic response cache.
pub async fn purge_cache(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OptionalPeer(peer): OptionalPeer,
) -> Response {
    if let Err(error) = authorize_admin(&state, &headers, peer) {
        return error.into_openai_response();
    }
    let evicted = state.cache().purge();
    Json(json!({
        "status": "purged",
        "evicted": evicted,
    }))
    .into_response()
}
