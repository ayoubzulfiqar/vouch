//! Per-backend circuit breaker.
//!
//! The breaker is a small `Closed -> Open -> HalfOpen -> Closed` state machine.
//! Failures are counted consecutively; a single success resets the count. When
//! the cooldown elapses, the breaker admits a bounded number of half-open
//! probes before deciding whether to close or re-open.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;

use crate::{config::ResilienceConfig, metrics};

/// Public circuit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    /// Normal operation.
    Closed,
    /// Cooldown active; requests are rejected.
    Open,
    /// Recovery probe in flight.
    HalfOpen,
}

impl CircuitState {
    /// Human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

/// Tuning values copied from the configuration on construction or reload.
#[derive(Debug, Clone)]
pub struct BreakerSettings {
    /// Consecutive failures required to open.
    pub failure_threshold: u32,
    /// Cooldown duration.
    pub reset_timeout: Duration,
    /// Concurrent probes admitted while half-open.
    pub half_open_max_probes: u32,
    /// Open immediately on HTTP 429.
    pub open_on_rate_limit: bool,
}

impl Default for BreakerSettings {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(30),
            half_open_max_probes: 1,
            open_on_rate_limit: true,
        }
    }
}

impl From<&ResilienceConfig> for BreakerSettings {
    fn from(config: &ResilienceConfig) -> Self {
        Self {
            failure_threshold: config.failure_threshold.max(1),
            reset_timeout: Duration::from_secs(config.reset_timeout_secs.max(1)),
            half_open_max_probes: config.half_open_max_probes.max(1),
            open_on_rate_limit: config.open_on_rate_limit,
        }
    }
}

/// Point-in-time view used by `/readyz` and `/v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct BreakerSnapshot {
    /// Current state.
    pub state: CircuitState,
    /// Consecutive failures observed.
    pub consecutive_failures: u32,
    /// Remaining cooldown in seconds when open.
    pub open_remaining_secs: u64,
    /// True when opened by an upstream authentication expiry/unauthorized error.
    pub auth_expired: bool,
    /// Last failure reason, if any.
    pub last_failure: Option<String>,
}

/// Classification metadata passed when recording a backend failure.
#[derive(Debug, Clone, Copy, Default)]
pub struct FailureDetails {
    /// True for HTTP 429 or CLI rate-limit output.
    pub rate_limited: bool,
    /// True for HTTP 401/403 or CLI auth-expiry output.
    pub auth_expired: bool,
    /// Optional explicit cooldown override (e.g. from HTTP `Retry-After`).
    pub retry_after: Option<Duration>,
}

#[derive(Debug)]
enum InternalState {
    Closed,
    Open { until: Instant },
    HalfOpen,
}

#[derive(Debug)]
struct Inner {
    state: InternalState,
    consecutive_failures: u32,
    half_open_probes: u32,
    auth_expired: bool,
    last_failure: Option<String>,
}

/// One breaker per backend.
#[derive(Debug)]
pub struct CircuitBreaker {
    backend: String,
    settings: Mutex<BreakerSettings>,
    inner: Mutex<Inner>,
}

impl CircuitBreaker {
    /// Create a closed breaker.
    pub fn new(backend: impl Into<String>, settings: BreakerSettings) -> Self {
        let backend = backend.into();
        metrics::set_circuit_state(&backend, 0);
        Self {
            backend,
            settings: Mutex::new(settings),
            inner: Mutex::new(Inner {
                state: InternalState::Closed,
                consecutive_failures: 0,
                half_open_probes: 0,
                auth_expired: false,
                last_failure: None,
            }),
        }
    }

    /// Backend identifier.
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Apply new tuning values without discarding the current state.
    pub fn reconfigure(&self, settings: BreakerSettings) {
        *self.settings.lock() = settings;
    }

    /// Ask whether a request may proceed.
    ///
    /// A `true` return for a half-open breaker consumes one probe slot. Callers
    /// must eventually call [`Self::record_success`] or
    /// [`Self::record_failure`].
    pub fn allow_request(&self) -> bool {
        let settings = self.settings.lock().clone();
        let mut transition = None;
        let allowed = {
            let mut inner = self.inner.lock();
            match &inner.state {
                InternalState::Closed => true,
                InternalState::Open { until } => {
                    if Instant::now() >= *until {
                        inner.state = InternalState::HalfOpen;
                        inner.half_open_probes = 1;
                        transition = Some(CircuitState::HalfOpen);
                        true
                    } else {
                        false
                    }
                }
                InternalState::HalfOpen => {
                    if inner.half_open_probes < settings.half_open_max_probes {
                        inner.half_open_probes += 1;
                        true
                    } else {
                        false
                    }
                }
            }
        };
        if let Some(state) = transition {
            self.publish(state);
        }
        allowed
    }

    /// Record a successful execution.
    pub fn record_success(&self) {
        let transition = {
            let mut inner = self.inner.lock();
            inner.consecutive_failures = 0;
            inner.half_open_probes = 0;
            inner.auth_expired = false;
            inner.last_failure = None;
            let was_closed = matches!(inner.state, InternalState::Closed);
            inner.state = InternalState::Closed;
            if was_closed {
                None
            } else {
                Some(CircuitState::Closed)
            }
        };
        if let Some(state) = transition {
            self.publish(state);
        }
    }

    /// Record a failure. `rate_limited` indicates an upstream HTTP 429.
    pub fn record_failure(&self, reason: impl Into<String>, rate_limited: bool) {
        self.record_failure_detailed(
            reason,
            FailureDetails {
                rate_limited,
                auth_expired: false,
                retry_after: None,
            },
        );
    }

    /// Record a failure with auth-expiry and dynamic cooldown metadata.
    pub fn record_failure_detailed(&self, reason: impl Into<String>, details: FailureDetails) {
        let reason = reason.into();
        let settings = self.settings.lock().clone();
        let transition = {
            let mut inner = self.inner.lock();
            inner.last_failure = Some(reason);
            if details.auth_expired {
                inner.auth_expired = true;
            }
            let should_open = match inner.state {
                InternalState::HalfOpen => true,
                _ if details.auth_expired => true,
                _ if details.rate_limited && settings.open_on_rate_limit => true,
                _ => {
                    inner.consecutive_failures = inner.consecutive_failures.saturating_add(1);
                    inner.consecutive_failures >= settings.failure_threshold
                }
            };
            if should_open {
                let cooldown = if details.rate_limited {
                    details.retry_after.unwrap_or(settings.reset_timeout)
                } else {
                    settings.reset_timeout
                };
                inner.state = InternalState::Open {
                    until: Instant::now() + cooldown.max(Duration::from_millis(1)),
                };
                inner.half_open_probes = 0;
                Some(CircuitState::Open)
            } else {
                None
            }
        };
        if let Some(state) = transition {
            self.publish(state);
        }
    }

    /// Record a neutral outcome (for example saturation) without changing the
    /// failure count. A half-open probe slot is returned to the pool.
    pub fn record_neutral(&self) {
        let mut inner = self.inner.lock();
        if matches!(inner.state, InternalState::HalfOpen) && inner.half_open_probes > 0 {
            inner.half_open_probes -= 1;
        }
    }

    /// Current state, transitioning an expired open circuit to half-open.
    pub fn state(&self) -> CircuitState {
        let mut transition = None;
        let state = {
            let mut inner = self.inner.lock();
            match &inner.state {
                InternalState::Closed => CircuitState::Closed,
                InternalState::Open { until } => {
                    if Instant::now() >= *until {
                        inner.state = InternalState::HalfOpen;
                        inner.half_open_probes = 0;
                        transition = Some(CircuitState::HalfOpen);
                        CircuitState::HalfOpen
                    } else {
                        CircuitState::Open
                    }
                }
                InternalState::HalfOpen => CircuitState::HalfOpen,
            }
        };
        if let Some(state) = transition {
            self.publish(state);
        }
        state
    }

    /// Detailed snapshot.
    pub fn snapshot(&self) -> BreakerSnapshot {
        let state = self.state();
        let inner = self.inner.lock();
        let open_remaining_secs = match inner.state {
            InternalState::Open { until } => {
                until.saturating_duration_since(Instant::now()).as_secs()
            }
            _ => 0,
        };
        BreakerSnapshot {
            state,
            consecutive_failures: inner.consecutive_failures,
            open_remaining_secs,
            auth_expired: inner.auth_expired,
            last_failure: inner.last_failure.clone(),
        }
    }

    fn publish(&self, state: CircuitState) {
        metrics::set_circuit_state(&self.backend, metrics::circuit_state_value(state));
        metrics::record_circuit_transition(&self.backend, state.as_str());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaker(threshold: u32, cooldown_ms: u64) -> CircuitBreaker {
        CircuitBreaker::new(
            "test",
            BreakerSettings {
                failure_threshold: threshold,
                reset_timeout: Duration::from_millis(cooldown_ms),
                half_open_max_probes: 1,
                open_on_rate_limit: true,
            },
        )
    }

    #[test]
    fn opens_after_threshold() {
        let breaker = breaker(2, 50);
        assert!(breaker.allow_request());
        breaker.record_failure("boom", false);
        assert_eq!(breaker.state(), CircuitState::Closed);
        breaker.record_failure("boom", false);
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.allow_request());
    }

    #[test]
    fn half_open_after_cooldown_and_closes_on_success() {
        let breaker = breaker(1, 20);
        breaker.record_failure("boom", false);
        assert_eq!(breaker.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        assert!(breaker.allow_request());
        assert!(!breaker.allow_request());
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn rate_limit_opens_immediately() {
        let breaker = breaker(10, 50);
        breaker.record_failure("429", true);
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn auth_expired_opens_immediately_and_clears_on_success() {
        let breaker = breaker(10, 10);
        breaker.record_failure_detailed(
            "please sign in",
            FailureDetails {
                rate_limited: false,
                auth_expired: true,
                retry_after: None,
            },
        );
        let snap = breaker.snapshot();
        assert_eq!(snap.state, CircuitState::Open);
        assert!(snap.auth_expired);

        std::thread::sleep(Duration::from_millis(20));
        assert!(breaker.allow_request());
        breaker.record_success();
        let snap = breaker.snapshot();
        assert_eq!(snap.state, CircuitState::Closed);
        assert!(!snap.auth_expired);
    }

    #[test]
    fn rate_limit_honors_retry_after_override() {
        let breaker = breaker(10, 10);
        breaker.record_failure_detailed(
            "429",
            FailureDetails {
                rate_limited: true,
                auth_expired: false,
                retry_after: Some(Duration::from_secs(3)),
            },
        );
        let snap = breaker.snapshot();
        assert_eq!(snap.state, CircuitState::Open);
        assert!(snap.open_remaining_secs >= 2);
    }
}
