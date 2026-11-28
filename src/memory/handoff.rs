//! SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic zero-LLM-token cross-agent handoff engine (`[shared_memory]`).
//!
//! Builds strictly BPE-budgeted `<vouch_shared_context>` envelopes from the
//! canonical [`SessionLedgerStore`] and model-free [`TraceBridge`] whenever execution
//! transitions across backends or initializes a new session.

use std::{sync::Arc, time::Duration};

use arc_swap::ArcSwap;

use crate::{
    config::{BackendConfig, RouteStrategy, SharedMemoryConfig, TraceBridgeConfig},
    driver::{ExecutionRequest, render_prompt},
    memory::{
        ledger::{
            HandoffReason, SessionLedgerEntry, SessionLedgerStore, extract_sanitized_repo_paths,
        },
        trace_bridge::TraceBridge,
    },
    telemetry,
    usage::count_tokens,
};

/// Result attached to an outbound `attempt` when context was hydrated for a handoff.
#[derive(Debug, Clone)]
pub struct HandoffPlan {
    /// Previous backend id when transitioning agents (`None` on fresh session grounding).
    pub handoff_from: Option<String>,
    /// Reason classification when transitioning agents.
    pub reason: Option<HandoffReason>,
    /// True when native CLI `delta_messages` / `backend_session_id` must be cleared.
    pub override_delta: bool,
}

/// Shared coordinator combining `[shared_memory]` ledger and `[trace]` bridge.
#[derive(Debug)]
pub struct SharedMemoryEngine {
    config: ArcSwap<SharedMemoryConfig>,
    ledger: Arc<SessionLedgerStore>,
    trace: Arc<TraceBridge>,
}

impl SharedMemoryEngine {
    /// Create a shared memory & trace engine.
    pub fn new(
        config: SharedMemoryConfig,
        ledger: Arc<SessionLedgerStore>,
        trace: Arc<TraceBridge>,
    ) -> Self {
        Self {
            config: ArcSwap::from_pointee(config),
            ledger,
            trace,
        }
    }

    /// Hot-reload configuration atomically.
    pub fn update_config(
        &self,
        config: SharedMemoryConfig,
        trace_cfg: TraceBridgeConfig,
        ws_cfg: crate::config::WorkspaceConfig,
        max_sessions: usize,
    ) {
        self.ledger
            .set_limits(max_sessions, config.max_turns_per_session);
        self.config.store(Arc::new(config));
        self.trace.update_config(trace_cfg, ws_cfg);
    }

    /// Canonical ledger handle.
    pub fn ledger(&self) -> &Arc<SessionLedgerStore> {
        &self.ledger
    }

    /// Model-free `trace` MCP bridge handle.
    pub fn trace(&self) -> &Arc<TraceBridge> {
        &self.trace
    }

    /// Current `[shared_memory]` config snapshot.
    pub fn config(&self) -> Arc<SharedMemoryConfig> {
        self.config.load_full()
    }

    /// Inspect ledger state + `trace` before invoking `candidate`, injecting
    /// `<vouch_shared_context>` cleanly into `attempt` on handoffs or new sessions.
    #[allow(clippy::too_many_arguments)]
    pub async fn hydrate_attempt_for_candidate(
        &self,
        base_request: &ExecutionRequest,
        candidate: &BackendConfig,
        attempt: &mut ExecutionRequest,
        previous_backend_failed: Option<&str>,
        route_contains_backend: impl Fn(&str) -> bool,
        strategy: RouteStrategy,
        ttl: Duration,
    ) -> HandoffPlan {
        let cfg = self.config.load();
        if !cfg.enabled {
            return HandoffPlan {
                handoff_from: None,
                reason: None,
                override_delta: false,
            };
        }

        let trace_cfg = self.trace.config();
        let entry = self.ledger.get(&base_request.session_key, ttl);
        let curr_backend = &candidate.id;

        if let Some(entry) = entry {
            let prev_backend = entry.active_backend.as_deref();
            let backend_changed = prev_backend.is_some_and(|prev| prev != curr_backend);
            let missing_native_session =
                attempt.backend_session_id.is_none() && !entry.turns.is_empty();

            if backend_changed {
                let prev = prev_backend.unwrap_or("unknown").to_string();
                let reason = classify_handoff_reason(
                    previous_backend_failed,
                    entry.last_model.as_deref(),
                    &base_request.model,
                    &prev,
                    &route_contains_backend,
                    strategy,
                );
                if cfg.inject_on_handoff {
                    let envelope = self
                        .build_handoff_envelope(
                            base_request,
                            &cfg,
                            &trace_cfg,
                            Some(&entry),
                            Some((&prev, curr_backend, reason)),
                        )
                        .await;
                    inject_envelope_into_attempt(attempt, &envelope);
                }
                attempt.delta_messages = None;
                attempt.backend_session_id = None;
                return HandoffPlan {
                    handoff_from: Some(prev),
                    reason: Some(reason),
                    override_delta: true,
                };
            }

            if missing_native_session
                && cfg.inject_on_handoff
                && (!candidate.resume_args.is_empty() || base_request.messages.len() <= 1)
            {
                let envelope = self
                    .build_handoff_envelope(base_request, &cfg, &trace_cfg, Some(&entry), None)
                    .await;
                inject_envelope_into_attempt(attempt, &envelope);
                attempt.delta_messages = None;
                return HandoffPlan {
                    handoff_from: None,
                    reason: None,
                    override_delta: true,
                };
            }

            return HandoffPlan {
                handoff_from: None,
                reason: None,
                override_delta: false,
            };
        }

        // Case B: Fresh session grounding when trace is enabled and `inject_on_new_session` is true.
        if cfg.inject_on_new_session && trace_cfg.enabled {
            let envelope = self
                .build_handoff_envelope(base_request, &cfg, &trace_cfg, None, None)
                .await;
            if !envelope.is_empty() {
                inject_envelope_into_attempt(attempt, &envelope);
            }
        }

        HandoffPlan {
            handoff_from: None,
            reason: None,
            override_delta: false,
        }
    }

    async fn build_handoff_envelope(
        &self,
        request: &ExecutionRequest,
        cfg: &SharedMemoryConfig,
        trace_cfg: &TraceBridgeConfig,
        entry: Option<&SessionLedgerEntry>,
        transition: Option<(&str, &str, HandoffReason)>,
    ) -> String {
        let model = &request.model;
        let max_total = cfg.max_context_tokens.max(1);
        let trace_budget = trace_cfg.max_trace_tokens.min(max_total);
        let ws_override = request
            .workspace_root
            .as_deref()
            .or(request.workdir.as_deref());
        let latest_user = latest_user_prompt(request);

        let mut candidate_files: Vec<String> =
            entry.map(|e| e.touched_files.clone()).unwrap_or_default();
        for rel in extract_sanitized_repo_paths(&latest_user, ws_override) {
            if !candidate_files.iter().any(|existing| existing == &rel) {
                candidate_files.push(rel);
            }
        }

        let trace_section = if trace_cfg.enabled && trace_budget > 0 {
            self.trace
                .gather_repo_context(
                    ws_override,
                    model,
                    &candidate_files,
                    &latest_user,
                    trace_budget,
                )
                .await
        } else {
            None
        };

        if entry.is_none() && trace_section.is_none() {
            return String::new();
        }

        let mut header_lines = vec![format!("session_id: {}", request.session_id_label())];
        if let Some((from_b, to_b, reason)) = transition {
            header_lines.push(format!(
                "handoff: {} -> {} (reason: {})",
                from_b,
                to_b,
                reason.as_str()
            ));
        } else {
            header_lines.push("mode: shared_context".into());
        }
        if let Some(entry) = entry {
            if !entry.checkpoints.is_empty() {
                header_lines.push(format!(
                    "checkpoints: {}",
                    entry
                        .checkpoints
                        .iter()
                        .rev()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join(" | ")
                ));
            }
            if !entry.touched_files.is_empty() {
                header_lines.push(format!("touched_files: {}", entry.touched_files.join(", ")));
            }
        }

        let header_block = telemetry::redact_text(&header_lines.join("\n"));
        let open_tag = "<vouch_shared_context>\n";
        let close_tag = "\n</vouch_shared_context>";
        let fixed_overhead =
            count_tokens(model, &format!("{open_tag}{header_block}{close_tag}")) as usize;

        let remaining_after_header = max_total.saturating_sub(fixed_overhead);
        let trace_cap = if entry.is_none() {
            remaining_after_header
        } else {
            remaining_after_header.min(trace_budget)
        };
        let clean_trace = trace_section
            .map(|t| truncate_to_bpe_budget(model, &t, trace_cap))
            .filter(|t| !t.trim().is_empty());

        let trace_tokens = clean_trace
            .as_deref()
            .map(|t| count_tokens(model, t) as usize + 2)
            .unwrap_or(0);
        let turns_budget = remaining_after_header.saturating_sub(trace_tokens);

        let turns_block = if let Some(entry) = entry
            && turns_budget > 12
            && !entry.turns.is_empty()
        {
            format_recent_turns_budgeted(model, entry, turns_budget)
        } else {
            String::new()
        };

        let mut inner_parts = vec![header_block];
        if !turns_block.is_empty() {
            inner_parts.push(format!("Prior Conversation & Actions:\n{turns_block}"));
        }
        if let Some(trace_text) = clean_trace {
            inner_parts.push(trace_text);
        }

        let mut body = inner_parts.join("\n\n");
        let full_overhead = count_tokens(model, &format!("{open_tag}{close_tag}")) as usize;
        if (count_tokens(model, &body) as usize) + full_overhead > max_total {
            let max_inner = max_total.saturating_sub(full_overhead);
            body = truncate_to_bpe_budget(model, &body, max_inner);
        }

        let mut envelope = format!("{open_tag}{body}{close_tag}");
        // Guarantee strict BPE bound `<= max_total` even if closing tag boundary shifted tokenization.
        while count_tokens(model, &envelope) as usize > max_total && !body.is_empty() {
            let new_len = floor_char_boundary(&body, body.len() * 7 / 8);
            if new_len == 0 || new_len == body.len() {
                body.clear();
            } else {
                body.truncate(new_len);
                body.push_str("...");
            }
            envelope = format!("{open_tag}{body}{close_tag}");
            if body.is_empty() {
                break;
            }
        }
        envelope
    }
}

fn classify_handoff_reason(
    previous_backend_failed: Option<&str>,
    last_model: Option<&str>,
    current_model: &str,
    prev_backend: &str,
    route_contains_backend: &impl Fn(&str) -> bool,
    _strategy: RouteStrategy,
) -> HandoffReason {
    if previous_backend_failed.is_some() {
        return HandoffReason::Failover;
    }
    if last_model != Some(current_model) || !route_contains_backend(prev_backend) {
        return HandoffReason::ModelSwitch;
    }
    HandoffReason::PolicyReorder
}

fn latest_user_prompt(request: &ExecutionRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_plain_text())
        .unwrap_or_else(|| request.prompt.clone())
}

fn inject_envelope_into_attempt(attempt: &mut ExecutionRequest, envelope: &str) {
    if envelope.is_empty() {
        return;
    }
    attempt.system = match attempt.system.take() {
        Some(existing) if !existing.trim().is_empty() => Some(format!("{existing}\n\n{envelope}")),
        _ => Some(envelope.to_string()),
    };
    attempt.prompt = render_prompt(attempt.system.as_deref(), &attempt.messages);
}

trait SessionLabelExt {
    fn session_id_label(&self) -> &str;
}

impl SessionLabelExt for ExecutionRequest {
    fn session_id_label(&self) -> &str {
        &self.session_key
    }
}

fn format_recent_turns_budgeted(
    model: &str,
    entry: &SessionLedgerEntry,
    max_tokens: usize,
) -> String {
    let mut lines_rev = Vec::new();
    let mut used = 0usize;
    for turn in entry.turns.iter().rev() {
        let mut line = format!("- [{}@{}]: {}", turn.role, turn.backend_id, turn.content);
        if let Some(tools) = &turn.tool_summary {
            line.push_str(&format!(" (tools: {tools})"));
        }
        let toks = count_tokens(model, &line) as usize + 1;
        if used + toks <= max_tokens {
            lines_rev.push(line);
            used += toks;
        } else if lines_rev.is_empty() {
            let trimmed = truncate_to_bpe_budget(model, &line, max_tokens);
            if !trimmed.is_empty() {
                lines_rev.push(trimmed);
            }
            break;
        } else {
            break;
        }
    }
    lines_rev.reverse();
    lines_rev.join("\n")
}

/// Binary-search truncate UTF-8 text cleanly to fit inside `max_tokens` BPE tokens.
pub fn truncate_to_bpe_budget(model: &str, text: &str, max_tokens: usize) -> String {
    if max_tokens == 0 || text.is_empty() {
        return String::new();
    }
    if count_tokens(model, text) as usize <= max_tokens {
        return text.to_string();
    }

    let suffix = "\n... [truncated]";
    let suffix_tokens = count_tokens(model, suffix) as usize;
    let target = max_tokens.saturating_sub(suffix_tokens);
    if target == 0 {
        let mut out = String::new();
        for ch in text.chars() {
            out.push(ch);
            if count_tokens(model, &out) as usize > max_tokens {
                out.pop();
                break;
            }
        }
        return out;
    }

    let char_offsets: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut lo = 0usize;
    let mut hi = char_offsets.len().saturating_sub(1);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let byte_idx = char_offsets[mid];
        if count_tokens(model, &text[..byte_idx]) as usize <= target {
            lo = mid;
        } else {
            hi = mid.saturating_sub(1);
        }
    }
    let mut keep_chars = lo;
    let mut out = format!("{}{suffix}", text[..char_offsets[keep_chars]].trim_end());
    while count_tokens(model, &out) as usize > max_tokens && keep_chars > 0 {
        keep_chars /= 2;
        out = format!("{}{suffix}", text[..char_offsets[keep_chars]].trim_end());
    }
    if count_tokens(model, &out) as usize > max_tokens {
        String::new()
    } else {
        out
    }
}

fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn truncate_to_bpe_budget_enforces_limit() {
        let text = "word ".repeat(600);
        let trimmed = truncate_to_bpe_budget("coding/default", &text, 80);
        assert!(count_tokens("coding/default", &trimmed) <= 80);
        assert!(trimmed.contains("[truncated]"));
    }

    #[test]
    fn sanitize_root_relative_helper_compiles_with_path() {
        let _ = Path::new(".");
    }
}
