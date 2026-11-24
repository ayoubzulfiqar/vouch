//! Prometheus instrumentation.
//!
//! The `metrics` facade is global by design. [`Metrics::install`] installs a
//! process-wide recorder once and returns a cheap handle used by `/metrics`.
//! Tests that build several gateways in one process share the recorder.

use std::sync::OnceLock;
use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::error::{GatewayError, Result};

/// Handle to the process-wide Prometheus recorder.
#[derive(Clone, Debug)]
pub struct Metrics {
    handle: Option<PrometheusHandle>,
}

static HANDLE: OnceLock<Option<PrometheusHandle>> = OnceLock::new();

impl Metrics {
    /// Install the recorder if it has not been installed yet.
    ///
    /// A failure to install (for example because a host application already
    /// installed a different recorder) degrades `/metrics` to a diagnostic
    /// comment rather than failing the gateway.
    pub fn install() -> Result<Self> {
        let handle = HANDLE
            .get_or_init(|| {
                let builder = PrometheusBuilder::new();
                match builder.set_buckets(&[
                    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                    30.0, 60.0, 120.0, 300.0,
                ]) {
                    Ok(builder) => builder.install_recorder().ok(),
                    Err(_) => None,
                }
            })
            .clone();
        Ok(Self { handle })
    }

    /// Render the current Prometheus exposition payload.
    pub fn render(&self) -> String {
        match &self.handle {
            Some(handle) => handle.render(),
            None => "# vouch: Prometheus recorder unavailable in this process\n".to_string(),
        }
    }

    /// Whether a recorder is active.
    pub fn is_active(&self) -> bool {
        self.handle.is_some()
    }
}

/// Record an ingress request.
pub fn record_request(protocol: &str, endpoint: &str, status: u16, elapsed: Duration) {
    metrics::counter!(
        "vouch_requests_total",
        "protocol" => protocol.to_string(),
        "endpoint" => endpoint.to_string(),
        "status" => status.to_string(),
    )
    .increment(1);
    metrics::histogram!(
        "vouch_request_duration_seconds",
        "protocol" => protocol.to_string(),
        "endpoint" => endpoint.to_string(),
    )
    .record(elapsed.as_secs_f64());
}

/// Record time spent waiting for a backend concurrency permit.
pub fn record_queue_wait(backend: &str, elapsed: Duration) {
    metrics::histogram!(
        "vouch_queue_wait_seconds",
        "backend" => backend.to_string(),
    )
    .record(elapsed.as_secs_f64());
}

/// Increment the active permit gauge for a backend.
pub fn inc_active_permit(backend: &str) {
    metrics::gauge!(
        "vouch_active_permits",
        "backend" => backend.to_string(),
    )
    .increment(1.0);
}

/// Decrement the active permit gauge for a backend.
pub fn dec_active_permit(backend: &str) {
    metrics::gauge!(
        "vouch_active_permits",
        "backend" => backend.to_string(),
    )
    .decrement(1.0);
}

/// Publish a circuit-breaker state (`0` closed, `1` half-open, `2` open).
pub fn set_circuit_state(backend: &str, state: u8) {
    metrics::gauge!(
        "vouch_circuit_breaker_state",
        "backend" => backend.to_string(),
    )
    .set(f64::from(state));
}

/// Count a circuit-breaker state transition.
pub fn record_circuit_transition(backend: &str, state: &str) {
    metrics::counter!(
        "vouch_circuit_breaker_transitions_total",
        "backend" => backend.to_string(),
        "state" => state.to_string(),
    )
    .increment(1);
}

/// Count a fallback from one backend to the next.
pub fn record_fallback(from: &str, to: &str) {
    metrics::counter!(
        "vouch_fallbacks_total",
        "from" => from.to_string(),
        "to" => to.to_string(),
    )
    .increment(1);
}

/// Record a child process exit. `code` is `"signal"` when terminated by one.
pub fn record_child_exit(backend: &str, code: &str) {
    metrics::counter!(
        "vouch_child_exit_total",
        "backend" => backend.to_string(),
        "code" => code.to_string(),
    )
    .increment(1);
}

/// Record a token rate-limit rejection.
pub fn record_token_rate_limited(token_name: &str) {
    metrics::counter!(
        "vouch_token_rate_limited_total",
        "token" => token_name.to_string(),
    )
    .increment(1);
}

/// Record an upstream health probe result.
pub fn record_health_probe(backend: &str, healthy: bool) {
    metrics::gauge!(
        "vouch_backend_health",
        "backend" => backend.to_string(),
    )
    .set(if healthy { 1.0 } else { 0.0 });
}

/// Convenience conversion used by the circuit breaker.
pub fn circuit_state_value(state: crate::router::circuit_breaker::CircuitState) -> u8 {
    match state {
        crate::router::circuit_breaker::CircuitState::Closed => 0,
        crate::router::circuit_breaker::CircuitState::HalfOpen => 1,
        crate::router::circuit_breaker::CircuitState::Open => 2,
    }
}

/// Error helper for callers that need to surface recorder failure.
pub fn unavailable() -> GatewayError {
    GatewayError::Internal("metrics recorder unavailable".into())
}
