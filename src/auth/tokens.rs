//! Virtual ingress token enforcement.
//!
//! When no `[[tokens]]` are configured the gateway runs in zero-friction local
//! mode and accepts any (or no) credential. Once tokens exist, every ingress
//! call is checked with a constant-time hash comparison, an optional sliding
//! one-minute RPM window, and a glob-based model allowlist.

use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use axum::http::HeaderMap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    config::{ModelAllowlist, TokenConfig},
    error::{GatewayError, Result},
    metrics,
};

const WINDOW: Duration = Duration::from_secs(60);

/// One configured credential.
#[derive(Debug)]
pub struct TokenPolicy {
    /// Label used in logs and metrics.
    pub name: String,
    /// SHA-256 of the presented secret.
    token_hash: [u8; 32],
    /// Optional requests-per-minute cap.
    pub rpm: Option<u32>,
    /// Compiled model allowlist; `None` means every model.
    allowlist: Option<ModelAllowlist>,
    window: Mutex<VecDeque<Instant>>,
}

impl TokenPolicy {
    /// Compile a token policy.
    pub fn new(config: &TokenConfig) -> Result<Self> {
        let hash = Sha256::digest(config.token.as_bytes());
        let mut token_hash = [0u8; 32];
        token_hash.copy_from_slice(&hash);
        let name = config
            .name
            .clone()
            .unwrap_or_else(|| format!("token-{}", hex::encode(&token_hash[..4])));
        let allowlist = if config.allowed_models.is_empty() {
            None
        } else {
            Some(ModelAllowlist::compile(&config.allowed_models)?)
        };
        Ok(Self {
            name,
            token_hash,
            rpm: config.rpm,
            allowlist,
            window: Mutex::new(VecDeque::new()),
        })
    }

    /// Whether this token may call `model`.
    pub fn allows_model(&self, model: &str) -> bool {
        self.allowlist
            .as_ref()
            .map(|allowlist| allowlist.matches(model))
            .unwrap_or(true)
    }

    fn check_rate_limit(&self) -> Result<()> {
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

    /// Replace all policies. On compile failure the previous set is kept.
    pub fn reload(&self, tokens: &[TokenConfig]) -> Result<()> {
        let mut policies = Vec::with_capacity(tokens.len());
        for token in tokens {
            policies.push(Arc::new(TokenPolicy::new(token)?));
        }
        self.policies.store(Arc::new(policies));
        Ok(())
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
        let principal = self.authenticate(headers)?;
        if !principal.allows_model(model) {
            return Err(GatewayError::Forbidden(format!(
                "token {:?} is not allowed to use model {model:?}",
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
            allowed_models: Vec::new(),
        }])
        .expect("registry");
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("vouch-secret-token"));
        assert!(registry.authenticate(&headers).is_ok());
    }
}
