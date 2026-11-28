//! SPDX-License-Identifier: GPL-3.0-or-later
//! Outbound secret and credential egress guardrails (`[security.egress_guard]`).
//!
//! Scans normalized prompts, chat messages, tool call arguments, and embedding
//! inputs before they leave the gateway to any HTTP upstream or CLI/PTY
//! subprocess. Supports three hot-reloadable modes: `off`, `redact`, and
//! `block` (returning HTTP `403`, OpenAI `code = "egress_policy_blocked"`).

use std::sync::{Arc, LazyLock};

use arc_swap::ArcSwap;
use regex::Regex;

use crate::{
    config::{EgressGuardConfig, EgressGuardMode},
    driver::{ContentPart, ExecutionRequest, MessageContent},
    error::{GatewayError, Result},
    metrics,
};

/// Replacement marker written in `mode = "redact"`.
pub const EGRESS_REDACTED_PLACEHOLDER: &str = "[REDACTED_EGRESS_SECRET]";

static BUILTIN_PRIVATE_KEY_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let patterns = [
        // PEM private key blocks and bare headers.
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
        // AWS access key IDs.
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        // GitHub personal access tokens and fine-grained PATs.
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        // Stripe live secret / restricted keys.
        r"\bsk_live_[A-Za-z0-9]{16,}\b",
    ];
    patterns
        .into_iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect()
});

#[derive(Debug)]
struct CompiledGuard {
    mode: EgressGuardMode,
    block_private_keys: bool,
    extra_patterns: Vec<Regex>,
}

impl CompiledGuard {
    fn compile(config: &EgressGuardConfig) -> Result<Self> {
        let mut extra_patterns = Vec::with_capacity(config.extra_patterns.len());
        for raw in &config.extra_patterns {
            let compiled = Regex::new(raw).map_err(|error| {
                GatewayError::Config(format!(
                    "security.egress_guard.extra_patterns invalid regex {raw:?}: {error}"
                ))
            })?;
            extra_patterns.push(compiled);
        }
        Ok(Self {
            mode: config.mode,
            block_private_keys: config.block_private_keys,
            extra_patterns,
        })
    }

    fn matches_any(&self, input: &str) -> bool {
        if input.is_empty() {
            return false;
        }
        if self.block_private_keys
            && BUILTIN_PRIVATE_KEY_PATTERNS
                .iter()
                .any(|re| re.is_match(input))
        {
            return true;
        }
        self.extra_patterns.iter().any(|re| re.is_match(input))
    }

    fn redact_in_place(&self, input: &mut String) -> usize {
        if input.is_empty() {
            return 0;
        }
        let mut count = 0usize;
        let mut current = std::mem::take(input);
        if self.block_private_keys {
            for re in BUILTIN_PRIVATE_KEY_PATTERNS.iter() {
                let hits = re.find_iter(&current).count();
                if hits > 0 {
                    count += hits;
                    current = re
                        .replace_all(&current, EGRESS_REDACTED_PLACEHOLDER)
                        .into_owned();
                }
            }
        }
        for re in &self.extra_patterns {
            let hits = re.find_iter(&current).count();
            if hits > 0 {
                count += hits;
                current = re
                    .replace_all(&current, EGRESS_REDACTED_PLACEHOLDER)
                    .into_owned();
            }
        }
        *input = current;
        count
    }
}

/// Hot-reloadable outbound egress guardrail engine.
#[derive(Debug)]
pub struct EgressGuard {
    inner: ArcSwap<CompiledGuard>,
}

impl EgressGuard {
    /// Construct from `[security.egress_guard]`.
    pub fn new(config: &EgressGuardConfig) -> Result<Self> {
        Ok(Self {
            inner: ArcSwap::from_pointee(CompiledGuard::compile(config)?),
        })
    }

    /// Atomically replace guardrail patterns and mode on config reload.
    pub fn update(&self, config: &EgressGuardConfig) -> Result<()> {
        let compiled = CompiledGuard::compile(config)?;
        self.inner.store(Arc::new(compiled));
        Ok(())
    }

    /// Current enforcement mode.
    pub fn mode(&self) -> EgressGuardMode {
        self.inner.load().mode
    }

    /// Inspect and apply configured guardrail actions across an outbound
    /// [`ExecutionRequest`] prior to spawning a backend or hitting an upstream.
    ///
    /// Returns the number of redacted occurrences when `mode = "redact"`,
    /// `Ok(0)` when clean or `mode = "off"`, and `Err(GatewayError::EgressBlocked)`
    /// when `mode = "block"` detects any secret pattern.
    pub fn inspect_request(&self, request: &mut ExecutionRequest) -> Result<usize> {
        let guard = self.inner.load();
        match guard.mode {
            EgressGuardMode::Off => Ok(0),
            EgressGuardMode::Block => {
                if Self::request_matches(&guard, request) {
                    metrics::record_egress_guard_action("blocked");
                    return Err(GatewayError::EgressBlocked(
                        "outbound payload contains sensitive credential or key material forbidden by egress_guard policy"
                            .into(),
                    ));
                }
                Ok(0)
            }
            EgressGuardMode::Redact => {
                let mut replacements = 0usize;
                if let Some(system) = &mut request.system {
                    replacements += guard.redact_in_place(system);
                }
                for msg in &mut request.messages {
                    replacements += Self::redact_message(&guard, msg);
                }
                if let Some(delta) = &mut request.delta_messages {
                    for msg in delta {
                        replacements += Self::redact_message(&guard, msg);
                    }
                }
                replacements += guard.redact_in_place(&mut request.prompt);
                if replacements > 0 {
                    metrics::record_egress_guard_action("redacted");
                }
                Ok(replacements)
            }
        }
    }

    /// Inspect or redact a list of plain strings (used by `/v1/embeddings`).
    pub fn inspect_texts(&self, inputs: &mut [String]) -> Result<usize> {
        let guard = self.inner.load();
        match guard.mode {
            EgressGuardMode::Off => Ok(0),
            EgressGuardMode::Block => {
                if inputs.iter().any(|s| guard.matches_any(s)) {
                    metrics::record_egress_guard_action("blocked");
                    return Err(GatewayError::EgressBlocked(
                        "embedding input contains sensitive credential or key material forbidden by egress_guard policy"
                            .into(),
                    ));
                }
                Ok(0)
            }
            EgressGuardMode::Redact => {
                let mut count = 0usize;
                for item in inputs {
                    count += guard.redact_in_place(item);
                }
                if count > 0 {
                    metrics::record_egress_guard_action("redacted");
                }
                Ok(count)
            }
        }
    }

    fn request_matches(guard: &CompiledGuard, request: &ExecutionRequest) -> bool {
        if request
            .system
            .as_deref()
            .is_some_and(|s| guard.matches_any(s))
        {
            return true;
        }
        if guard.matches_any(&request.prompt) {
            return true;
        }
        request
            .messages
            .iter()
            .any(|msg| Self::message_matches(guard, msg))
            || request
                .delta_messages
                .as_ref()
                .is_some_and(|delta| delta.iter().any(|msg| Self::message_matches(guard, msg)))
    }

    fn message_matches(guard: &CompiledGuard, msg: &crate::driver::ChatMessage) -> bool {
        let content_hit = match &msg.content {
            MessageContent::Text(t) => guard.matches_any(t),
            MessageContent::Parts(parts) => parts.iter().any(|p| match p {
                ContentPart::Text { text, .. } => guard.matches_any(text),
                ContentPart::ImageUrl { url } => guard.matches_any(url),
                ContentPart::ImageBase64 { .. } => false,
            }),
        };
        if content_hit {
            return true;
        }
        msg.tool_calls
            .iter()
            .any(|call| guard.matches_any(&call.arguments))
    }

    fn redact_message(guard: &CompiledGuard, msg: &mut crate::driver::ChatMessage) -> usize {
        let mut hits = 0usize;
        match &mut msg.content {
            MessageContent::Text(text) => {
                hits += guard.redact_in_place(text);
            }
            MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text, .. } => {
                            hits += guard.redact_in_place(text);
                        }
                        ContentPart::ImageUrl { url } => {
                            hits += guard.redact_in_place(url);
                        }
                        ContentPart::ImageBase64 { .. } => {}
                    }
                }
            }
        }
        for call in &mut msg.tool_calls {
            hits += guard.redact_in_place(&mut call.arguments);
        }
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{ChatMessage, WorkspaceMode};
    use std::collections::BTreeMap;
    use tokio_util::sync::CancellationToken;

    fn sample_request(text: &str) -> ExecutionRequest {
        ExecutionRequest {
            request_id: "req-1".into(),
            session_key: "sess-1".into(),
            model: "coding/default".into(),
            system: None,
            messages: vec![ChatMessage::text("user", text)],
            prompt: text.into(),
            stream: false,
            temperature: None,
            max_tokens: None,
            workspace_mode: WorkspaceMode::Auto,
            workdir: None,
            workspace_root: None,
            cancel: CancellationToken::new(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            backend_session_id: None,
            delta_messages: None,
            session_capture: Arc::new(parking_lot::Mutex::new(None)),
            trace_context: None,
            profile_id: None,
            profile_env: BTreeMap::new(),
            allowed_backends: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn redacts_builtin_and_custom_egress_secrets() {
        let guard = EgressGuard::new(&EgressGuardConfig {
            mode: EgressGuardMode::Redact,
            block_private_keys: true,
            extra_patterns: vec![r"CUSTOM_TOKEN_[0-9]+".into()],
        })
        .expect("guard");
        let mut req = sample_request(
            "deploy key AKIAIOSFODNN7EXAMPLE and CUSTOM_TOKEN_42 and -----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
        );
        let hits = guard.inspect_request(&mut req).expect("redacted ok");
        assert_eq!(hits, 6);
        assert!(
            req.prompt.contains(EGRESS_REDACTED_PLACEHOLDER)
                && !req.prompt.contains("AKIAIOSFODNN7EXAMPLE")
                && !req.prompt.contains("CUSTOM_TOKEN_42")
                && !req.prompt.contains("BEGIN RSA PRIVATE KEY")
        );
    }

    #[test]
    fn blocks_request_with_403_egress_policy_blocked() {
        let guard = EgressGuard::new(&EgressGuardConfig {
            mode: EgressGuardMode::Block,
            block_private_keys: true,
            extra_patterns: Vec::new(),
        })
        .expect("guard");
        let mut req = sample_request("token is sk_live_51H7xExampleSecretValue");
        let err = guard.inspect_request(&mut req).expect_err("must block");
        assert!(matches!(err, GatewayError::EgressBlocked(_)));
        assert_eq!(err.status(), axum::http::StatusCode::FORBIDDEN);
    }
}
