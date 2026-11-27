//! Unified gateway error type.
//!
//! Every subsystem returns [`GatewayError`]. The ingress adapters convert it
//! into either the OpenAI or the Anthropic error envelope, so clients always
//! receive a protocol-shaped error instead of an axum rejection page.

use std::time::Duration;

use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Errors surfaced by the gateway.
///
/// Variants are intentionally coarse: they map to a small, stable set of HTTP
/// status codes and protocol error types. Contextual detail belongs in the
/// message, never in a new variant that downstream code must understand.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    /// Invalid or unreadable configuration.
    #[error("configuration error: {0}")]
    Config(String),

    /// The client sent a syntactically valid request that cannot be served.
    #[error("invalid request: {0}")]
    BadRequest(String),

    /// A credential is missing or invalid.
    #[error("authentication required: {0}")]
    Unauthorized(String),

    /// The credential is valid but the request is outside its scope.
    #[error("forbidden: {0}")]
    Forbidden(String),

    /// The requested model, route, or resource does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// The caller exceeded a virtual token quota.
    #[error("rate limit exceeded: {message}")]
    RateLimited {
        /// Suggested number of seconds before retrying.
        retry_after_secs: u64,
        /// Human-readable explanation.
        message: String,
    },

    /// Every candidate backend was at its concurrency limit.
    #[error("all backends are saturated: {0}")]
    Saturated(String),

    /// No backend is currently eligible (all circuits are open).
    #[error("no healthy backend available: {0}")]
    NoHealthyBackend(String),

    /// An upstream daemon or CLI returned an error.
    #[error("upstream error: {0}")]
    Upstream(String),

    /// An upstream daemon rate-limited the gateway.
    #[error("upstream rate limited: {message}")]
    UpstreamRateLimited {
        /// Parsed `Retry-After` duration in seconds when provided.
        retry_after_secs: Option<u64>,
        /// Redacted upstream detail.
        message: String,
    },

    /// An upstream backend rejected credentials or reported an expired session.
    #[error("upstream authentication expired: {0}")]
    UpstreamAuthExpired(String),

    /// A backend exceeded its deadline.
    #[error("backend timed out after {0:?}")]
    Timeout(Duration),

    /// The downstream client went away.
    #[error("client disconnected")]
    ClientDisconnected,

    /// Local I/O failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// An invariant was violated or a component failed unexpectedly.
    #[error("internal error: {0}")]
    Internal(String),
}

impl GatewayError {
    /// Convenience constructor for upstream HTTP 429 or CLI rate limits.
    pub fn upstream_rate_limited(
        message: impl Into<String>,
        retry_after_secs: Option<u64>,
    ) -> Self {
        Self::UpstreamRateLimited {
            retry_after_secs,
            message: message.into(),
        }
    }

    /// Extract the parsed `Retry-After` duration when this error is an upstream 429.
    pub fn upstream_rate_limit_retry_after(&self) -> Option<Duration> {
        match self {
            Self::UpstreamRateLimited {
                retry_after_secs: Some(secs),
                ..
            } => Some(Duration::from_secs((*secs).max(1))),
            _ => None,
        }
    }

    /// True when this error represents an expired or invalid upstream credential.
    pub fn is_auth_expired(&self) -> bool {
        matches!(self, Self::UpstreamAuthExpired(_))
    }

    /// True when this error represents an upstream rate limit.
    pub fn is_rate_limited(&self) -> bool {
        matches!(self, Self::UpstreamRateLimited { .. })
    }

    /// HTTP status code for this error.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Config(_) | Self::Internal(_) | Self::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Forbidden(_) => StatusCode::FORBIDDEN,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::RateLimited { .. } | Self::UpstreamRateLimited { .. } => {
                StatusCode::TOO_MANY_REQUESTS
            }
            Self::Saturated(_) | Self::NoHealthyBackend(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Upstream(_) | Self::UpstreamAuthExpired(_) => StatusCode::BAD_GATEWAY,
            Self::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            Self::ClientDisconnected => StatusCode::REQUEST_TIMEOUT,
        }
    }

    /// Retry hint, when the error is retryable.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimited {
                retry_after_secs, ..
            } => Some(*retry_after_secs),
            Self::UpstreamRateLimited {
                retry_after_secs, ..
            } => Some(retry_after_secs.unwrap_or(5)),
            Self::Saturated(_) | Self::NoHealthyBackend(_) => Some(2),
            _ => None,
        }
    }

    /// Whether the router should try the next backend in the chain.
    ///
    /// Client mistakes are final; saturation and backend failures are
    /// retryable against another candidate.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Saturated(_)
                | Self::NoHealthyBackend(_)
                | Self::Upstream(_)
                | Self::UpstreamRateLimited { .. }
                | Self::UpstreamAuthExpired(_)
                | Self::Timeout(_)
                | Self::Io(_)
                | Self::Internal(_)
        )
    }

    /// Whether this error should count against a backend's circuit breaker.
    ///
    /// Client errors and saturation are not backend health signals.
    pub fn counts_as_backend_failure(&self) -> bool {
        matches!(
            self,
            Self::Upstream(_)
                | Self::UpstreamRateLimited { .. }
                | Self::UpstreamAuthExpired(_)
                | Self::Timeout(_)
                | Self::Io(_)
                | Self::Internal(_)
        )
    }

    /// OpenAI error `type` field.
    fn openai_error_type(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request_error",
            Self::Unauthorized(_) | Self::UpstreamAuthExpired(_) => "authentication_error",
            Self::Forbidden(_) => "permission_error",
            Self::NotFound(_) => "not_found_error",
            Self::RateLimited { .. } | Self::UpstreamRateLimited { .. } => "rate_limit_error",
            Self::Saturated(_) | Self::NoHealthyBackend(_) => "service_unavailable_error",
            Self::Config(_) | Self::Internal(_) | Self::Io(_) => "server_error",
            Self::Upstream(_) | Self::Timeout(_) | Self::ClientDisconnected => "api_error",
        }
    }

    /// Stable machine-readable OpenAI error code.
    fn openai_code(&self) -> &'static str {
        match self {
            Self::Config(_) => "configuration_error",
            Self::BadRequest(_) => "invalid_request",
            Self::Unauthorized(_) => "invalid_api_key",
            Self::Forbidden(_) => "model_not_allowed",
            Self::NotFound(_) => "model_not_found",
            Self::RateLimited { .. } => "rate_limit_exceeded",
            Self::UpstreamRateLimited { .. } => "upstream_rate_limited",
            Self::UpstreamAuthExpired(_) => "upstream_auth_expired",
            Self::Saturated(_) => "backend_saturated",
            Self::NoHealthyBackend(_) => "no_healthy_backend",
            Self::Upstream(_) => "upstream_error",
            Self::Timeout(_) => "upstream_timeout",
            Self::ClientDisconnected => "client_disconnected",
            Self::Io(_) | Self::Internal(_) => "internal_error",
        }
    }

    /// Anthropic error `type` field.
    fn anthropic_error_type(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request_error",
            Self::Unauthorized(_) | Self::UpstreamAuthExpired(_) => "authentication_error",
            Self::Forbidden(_) => "permission_error",
            Self::NotFound(_) => "not_found_error",
            Self::RateLimited { .. } | Self::UpstreamRateLimited { .. } => "rate_limit_error",
            Self::Saturated(_) | Self::NoHealthyBackend(_) => "overloaded_error",
            Self::Upstream(_) | Self::Timeout(_) => "api_error",
            Self::ClientDisconnected => "api_error",
            Self::Config(_) | Self::Internal(_) | Self::Io(_) => "api_error",
        }
    }

    /// Render an OpenAI-compatible error response.
    pub fn into_openai_response(self) -> Response {
        let status = self.status();
        let body = json!({
            "error": {
                "message": self.to_string(),
                "type": self.openai_error_type(),
                "param": null,
                "code": self.openai_code(),
            }
        });
        self.response_with_headers(status, body)
    }

    /// Render an Anthropic-compatible error response.
    pub fn into_anthropic_response(self) -> Response {
        let status = self.status();
        let body = json!({
            "type": "error",
            "error": {
                "type": self.anthropic_error_type(),
                "message": self.to_string(),
            }
        });
        self.response_with_headers(status, body)
    }

    fn response_with_headers(self, status: StatusCode, body: serde_json::Value) -> Response {
        let retry_after = self.retry_after_secs();
        let mut response = (status, Json(body)).into_response();
        if let Some(secs) = retry_after
            && let Ok(value) = HeaderValue::from_str(&secs.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

impl From<reqwest::Error> for GatewayError {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout(Duration::from_secs(0))
        } else {
            Self::Upstream(format!("http transport error: {error}"))
        }
    }
}

impl From<serde_json::Error> for GatewayError {
    fn from(error: serde_json::Error) -> Self {
        Self::BadRequest(format!("invalid JSON: {error}"))
    }
}

impl From<tokio::time::error::Elapsed> for GatewayError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        Self::Timeout(Duration::from_secs(0))
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        self.into_openai_response()
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T, E = GatewayError> = std::result::Result<T, E>;
