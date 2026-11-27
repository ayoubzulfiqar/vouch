//! Virtual ingress token enforcement.
//!
//! When no `[[tokens]]` are configured the gateway runs in zero-friction local
//! mode and accepts any (or no) credential. Once tokens exist, every ingress
//! call is checked with a constant-time hash comparison, an optional sliding
//! one-minute RPM window, and a glob-based model allowlist.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwap;
use axum::http::HeaderMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    config::{ModelAllowlist, TokenConfig},
    error::{GatewayError, Result},
    metrics,
};

const WINDOW: Duration = Duration::from_secs(60);
const SECS_PER_DAY: u64 = 86_400;

/// Rolling UTC-day BPE token counter persisted by token SHA-256 hex digest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyTokenCounter {
    /// UTC day number (`unix_secs / 86400`).
    pub utc_day: u64,
    /// Cumulative prompt + completion BPE tokens consumed on `utc_day`.
    pub tokens: u64,
}

/// Snapshot of one configured token's daily budget status (zero raw secrets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudgetStatus {
    /// Safe token display name.
    pub name: String,
    /// Configured daily BPE budget, if any.
    pub daily_token_budget: Option<u64>,
    /// Tokens consumed today (UTC).
    pub daily_tokens_used: u64,
}

fn utc_day_for(now: SystemTime) -> u64 {
    now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() / SECS_PER_DAY
}

fn seconds_until_utc_midnight(now: SystemTime) -> u64 {
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    (SECS_PER_DAY - (secs % SECS_PER_DAY)).max(1)
}

/// One configured credential.
#[derive(Debug)]
pub struct TokenPolicy {
    /// Label used in logs and metrics.
    pub name: String,
    /// SHA-256 of the presented secret.
    token_hash: [u8; 32],
    /// Hex digest used purely as an anonymous persistence key.
    hash_hex: String,
    /// Optional requests-per-minute cap.
    pub rpm: Option<u32>,
    /// Optional rolling UTC day BPE token cap.
    pub daily_token_budget: Option<u64>,
    /// Compiled model allowlist; `None` means every model.
    allowlist: Option<ModelAllowlist>,
    window: Mutex<VecDeque<Instant>>,
    daily_usage: Mutex<DailyTokenCounter>,
}

impl TokenPolicy {
    /// Compile a token policy.
    pub fn new(config: &TokenConfig) -> Result<Self> {
        let hash = Sha256::digest(config.token.as_bytes());
        let mut token_hash = [0u8; 32];
        token_hash.copy_from_slice(&hash);
        let hash_hex = hex::encode(token_hash);
        let name = config
            .name
            .clone()
            .unwrap_or_else(|| format!("token-{}", &hash_hex[..8]));
        let allowlist = if config.allowed_models.is_empty() {
            None
        } else {
            Some(ModelAllowlist::compile(&config.allowed_models)?)
        };
        Ok(Self {
            name,
            token_hash,
            hash_hex,
            rpm: config.rpm,
            daily_token_budget: config.daily_token_budget,
            allowlist,
            window: Mutex::new(VecDeque::new()),
            daily_usage: Mutex::new(DailyTokenCounter {
                utc_day: utc_day_for(SystemTime::now()),
                tokens: 0,
            }),
        })
    }

    /// Whether this token may call `model`.
    pub fn allows_model(&self, model: &str) -> bool {
        self.allowlist
            .as_ref()
            .map(|allowlist| allowlist.matches(model))
            .unwrap_or(true)
    }

    /// Consume daily BPE budget quota before checking RPM so rejected calls do not burn RPM.
    fn check_daily_budget(&self) -> Result<()> {
        let Some(budget) = self.daily_token_budget else {
            return Ok(());
        };
        let now = SystemTime::now();
        let today = utc_day_for(now);
        let mut daily = self.daily_usage.lock();
        if daily.utc_day != today {
            daily.utc_day = today;
            daily.tokens = 0;
        }
        if daily.tokens >= budget {
            let retry_after_secs = seconds_until_utc_midnight(now);
            metrics::record_token_rate_limited(&self.name);
            return Err(GatewayError::DailyTokenBudgetExceeded {
                retry_after_secs,
                message: format!(
                    "token {:?} exceeded its daily BPE budget of {budget} tokens ({} used today)",
                    self.name, daily.tokens
                ),
            });
        }
        Ok(())
    }

    /// Record consumed prompt + completion BPE tokens against the current UTC day.
    pub fn record_tokens(&self, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let today = utc_day_for(SystemTime::now());
        let mut daily = self.daily_usage.lock();
        if daily.utc_day != today {
            daily.utc_day = today;
            daily.tokens = 0;
        }
        daily.tokens = daily.tokens.saturating_add(tokens);
    }

    /// Snapshot current UTC day token consumption.
    pub fn daily_counter(&self) -> DailyTokenCounter {
        let today = utc_day_for(SystemTime::now());
        let mut daily = self.daily_usage.lock();
        if daily.utc_day != today {
            daily.utc_day = today;
            daily.tokens = 0;
        }
        *daily
    }

    fn set_daily_counter_if_today(&self, saved: DailyTokenCounter) {
        let today = utc_day_for(SystemTime::now());
        if saved.utc_day != today {
            return;
        }
        let mut daily = self.daily_usage.lock();
        if daily.utc_day != today {
            *daily = saved;
        } else {
            daily.tokens = daily.tokens.max(saved.tokens);
        }
    }

    fn check_rate_limit(&self) -> Result<()> {
        self.check_daily_budget()?;
        let Some(rpm) = self.rpm else {
            return Ok(());
        };
        let now = Instant::now();
        let mut window = self.window.lock();
        while let Some(front) = window.front() {
            if now.duration_since(*front) >= WINDOW {
                window.pop_front();
            } else {
                break;
            }
        }
        if window.len() >= rpm as usize {
            let retry_after_secs = window
                .front()
                .map(|front| WINDOW.saturating_sub(now.duration_since(*front)).as_secs())
                .unwrap_or(1)
                .max(1);
            metrics::record_token_rate_limited(&self.name);
            return Err(GatewayError::RateLimited {
                retry_after_secs,
                message: format!(
                    "token {:?} exceeded its {} requests-per-minute limit",
                    self.name, rpm
                ),
            });
        }
        window.push_back(now);
        Ok(())
    }
}

/// Authenticated caller identity.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Display name.
    pub name: String,
    /// True in zero-friction local mode.
    pub anonymous: bool,
    policy: Option<Arc<TokenPolicy>>,
}

impl Principal {
    /// Anonymous local principal.
    pub fn local() -> Self {
        Self {
            name: "local".into(),
            anonymous: true,
            policy: None,
        }
    }

    /// Whether this caller may use `model`.
    pub fn allows_model(&self, model: &str) -> bool {
        self.policy
            .as_ref()
            .map(|policy| policy.allows_model(model))
            .unwrap_or(true)
    }

    /// Whether this caller may use either the requested model/alias or resolved canonical route.
    pub fn allows_model_or_resolved(&self, requested: &str, resolved: &str) -> bool {
        self.allows_model(requested) || self.allows_model(resolved)
    }

    /// Record completed prompt + completion BPE tokens against this caller's daily budget.
    pub fn record_tokens(&self, tokens: u64) {
        if let Some(policy) = &self.policy {
            policy.record_tokens(tokens);
        }
    }

    /// Whether the caller is authenticated by a configured token.
    pub fn is_authenticated(&self) -> bool {
        self.policy.is_some()
    }
}

/// Token registry with atomic reload support.
#[derive(Debug)]
pub struct TokenRegistry {
    policies: ArcSwap<Vec<Arc<TokenPolicy>>>,
}

impl TokenRegistry {
    /// Build a registry from configuration.
    pub fn new(tokens: &[TokenConfig]) -> Result<Self> {
        let registry = Self {
            policies: ArcSwap::from_pointee(Vec::new()),
        };
        registry.reload(tokens)?;
        Ok(registry)
    }

    /// Replace all policies, preserving today's BPE usage counters for matching token hashes.
    pub fn reload(&self, tokens: &[TokenConfig]) -> Result<()> {
        let prior = self.export_daily_state();
        let mut policies = Vec::with_capacity(tokens.len());
        for token in tokens {
            let policy = TokenPolicy::new(token)?;
            if let Some(saved) = prior.get(&policy.hash_hex) {
                policy.set_daily_counter_if_today(*saved);
            }
            policies.push(Arc::new(policy));
        }
        self.policies.store(Arc::new(policies));
        Ok(())
    }

    /// Snapshot current UTC-day token counters keyed by anonymous SHA-256 hex digest.
    pub fn export_daily_state(&self) -> BTreeMap<String, DailyTokenCounter> {
        let mut map = BTreeMap::new();
        for policy in self.policies.load().iter() {
            let counter = policy.daily_counter();
            if counter.tokens > 0 {
                map.insert(policy.hash_hex.clone(), counter);
            }
        }
        map
    }

    /// Restore previously persisted daily token counters matching today's UTC date.
    pub fn restore_daily_state(&self, saved: &BTreeMap<String, DailyTokenCounter>) {
        for policy in self.policies.load().iter() {
            if let Some(counter) = saved.get(&policy.hash_hex) {
                policy.set_daily_counter_if_today(*counter);
            }
        }
    }

    /// Summary of configured token daily budgets (safe to expose without secrets).
    pub fn budget_statuses(&self) -> Vec<TokenBudgetStatus> {
        self.policies
            .load()
            .iter()
            .map(|policy| TokenBudgetStatus {
                name: policy.name.clone(),
                daily_token_budget: policy.daily_token_budget,
                daily_tokens_used: policy.daily_counter().tokens,
            })
            .collect()
    }

    /// Whether token enforcement is active.
    pub fn enabled(&self) -> bool {
        !self.policies.load().is_empty()
    }

    /// Authenticate without checking a model.
    pub fn authenticate(&self, headers: &HeaderMap) -> Result<Principal> {
        let policies = self.policies.load_full();
        if policies.is_empty() {
            return Ok(Principal::local());
        }
        let presented = extract_credential(headers).ok_or_else(|| {
            GatewayError::Unauthorized(
                "missing API key: send Authorization: Bearer <token> or x-api-key".into(),
            )
        })?;
        let presented_digest = Sha256::digest(presented.as_bytes());
        let mut presented_hash = [0u8; 32];
        presented_hash.copy_from_slice(&presented_digest);
        let mut matched: Option<Arc<TokenPolicy>> = None;
        // Iterate all policies so timing does not reveal the matching index.
        for policy in policies.iter() {
            if policy.token_hash.ct_eq(&presented_hash).into() {
                matched = Some(policy.clone());
            }
        }
        let policy = matched.ok_or_else(|| GatewayError::Unauthorized("invalid API key".into()))?;
        policy.check_rate_limit()?;
        Ok(Principal {
            name: policy.name.clone(),
            anonymous: false,
            policy: Some(policy),
        })
    }

    /// Authenticate and enforce the token's model allowlist.
    pub fn authorize(&self, headers: &HeaderMap, model: &str) -> Result<Principal> {
        self.authorize_route(headers, model, model)
    }

    /// Authenticate and enforce `allowed_models` against both requested alias/model and resolved route.
    pub fn authorize_route(
        &self,
        headers: &HeaderMap,
        requested: &str,
        resolved: &str,
    ) -> Result<Principal> {
        let principal = self.authenticate(headers)?;
        if !principal.allows_model_or_resolved(requested, resolved) {
            return Err(GatewayError::Forbidden(format!(
                "token {:?} is not allowed to use model {requested:?}",
                principal.name
            )));
        }
        Ok(principal)
    }
}

fn extract_credential(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        let value = value.trim();
        if let Some(token) = value
            .strip_prefix("Bearer ")
            .or_else(|| value.strip_prefix("bearer "))
        {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
        // Some local clients send the raw token without a scheme.
        if !value.is_empty() && !value.contains(' ') {
            return Some(value.to_string());
        }
    }
    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).expect("header"),
        );
        headers
    }

    #[test]
    fn local_mode_accepts_anything() {
        let registry = TokenRegistry::new(&[]).expect("registry");
        assert!(registry.authenticate(&HeaderMap::new()).is_ok());
        assert!(
            registry
                .authorize(&headers("anything"), "any/model")
                .is_ok()
        );
    }

    #[test]
    fn enforces_token_and_model_scope() {
        let registry = TokenRegistry::new(&[TokenConfig {
            token: "vouch-secret-token".into(),
            name: Some("test".into()),
            rpm: None,
            daily_token_budget: None,
            allowed_models: vec!["coding/*".into()],
        }])
        .expect("registry");
        assert!(registry.authenticate(&HeaderMap::new()).is_err());
        assert!(
            registry
                .authenticate(&headers("wrong-token-value"))
                .is_err()
        );
        assert!(
            registry
                .authorize(&headers("vouch-secret-token"), "coding/default")
                .is_ok()
        );
        assert!(
            registry
                .authorize_route(
                    &headers("vouch-secret-token"),
                    "fast-coder",
                    "coding/default"
                )
                .is_ok()
        );
        assert!(
            registry
                .authorize(&headers("vouch-secret-token"), "hermes/agent")
                .is_err()
        );
    }

    #[test]
    fn enforces_rpm() {
        let registry = TokenRegistry::new(&[TokenConfig {
            token: "vouch-secret-token".into(),
            name: Some("test".into()),
            rpm: Some(2),
            daily_token_budget: None,
            allowed_models: Vec::new(),
        }])
        .expect("registry");
        assert!(
            registry
                .authenticate(&headers("vouch-secret-token"))
                .is_ok()
        );
        assert!(
            registry
                .authenticate(&headers("vouch-secret-token"))
                .is_ok()
        );
        let error = registry
            .authenticate(&headers("vouch-secret-token"))
            .expect_err("must rate limit");
        assert!(matches!(error, GatewayError::RateLimited { .. }));
    }

    #[test]
    fn accepts_x_api_key() {
        let registry = TokenRegistry::new(&[TokenConfig {
            token: "vouch-secret-token".into(),
            name: None,
            rpm: None,
            daily_token_budget: None,
            allowed_models: Vec::new(),
        }])
        .expect("registry");
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("vouch-secret-token"));
        assert!(registry.authenticate(&headers).is_ok());
    }

    #[test]
    fn enforces_daily_token_budget_and_preserves_across_reload() {
        let cfg = TokenConfig {
            token: "vouch-secret-budget".into(),
            name: Some("budgeted".into()),
            rpm: None,
            daily_token_budget: Some(100),
            allowed_models: Vec::new(),
        };
        let registry = TokenRegistry::new(std::slice::from_ref(&cfg)).expect("registry");
        let p = registry
            .authenticate(&headers("vouch-secret-budget"))
            .expect("initial auth ok");
        p.record_tokens(120);
        // Reload keeps daily usage counter for unchanged token hash.
        registry.reload(std::slice::from_ref(&cfg)).expect("reload");
        let err = registry
            .authenticate(&headers("vouch-secret-budget"))
            .expect_err("daily budget must block");
        assert!(matches!(
            err,
            GatewayError::DailyTokenBudgetExceeded {
                retry_after_secs,
                ..
            } if (1..=86_400).contains(&retry_after_secs)
        ));
    }
}
