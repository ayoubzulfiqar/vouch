//! SPDX-License-Identifier: GPL-3.0-or-later
//! Configuration inspection & structural diff tooling (`vouch config diff`).

use std::{collections::BTreeMap, io::Write};

use serde::Serialize;

use crate::{
    cli::{ConfigArgs, ConfigCommand, ConfigDiffArgs},
    config::{BackendConfig, GatewayConfig, RoutePolicyConfig, TokenConfig},
    error::Result,
};

/// Structured semantic summary of differences between two validated configs.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ConfigDiffReport {
    /// Server-level setting changes.
    pub server_changes: Vec<String>,
    /// Backend additions, removals, and field deltas.
    pub backend_changes: Vec<String>,
    /// Route chain additions, removals, and updates.
    pub route_changes: Vec<String>,
    /// Alias additions, removals, and updates.
    pub alias_changes: Vec<String>,
    /// Route policy strategy / hedge delay deltas.
    pub policy_changes: Vec<String>,
    /// Token quota and permission deltas (raw tokens redacted).
    pub token_changes: Vec<String>,
    /// Security & flight recorder changes.
    pub feature_changes: Vec<String>,
}

impl ConfigDiffReport {
    /// True if zero semantic differences exist between the two configurations.
    pub fn is_empty(&self) -> bool {
        self.server_changes.is_empty()
            && self.backend_changes.is_empty()
            && self.route_changes.is_empty()
            && self.alias_changes.is_empty()
            && self.policy_changes.is_empty()
            && self.token_changes.is_empty()
            && self.feature_changes.is_empty()
    }
}

/// Dispatch `vouch config ...` subcommands.
pub fn run(args: &ConfigArgs) -> Result<()> {
    match &args.command {
        ConfigCommand::Diff(diff_args) => run_diff(diff_args),
    }
}

fn run_diff(args: &ConfigDiffArgs) -> Result<()> {
    let current = GatewayConfig::load(&args.current)?;
    let candidate = GatewayConfig::load(&args.candidate)?;
    let report = compute_diff(&current, &candidate);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(
        stdout,
        "vouch config diff: {} -> {}",
        args.current.display(),
        args.candidate.display()
    );
    if report.is_empty() {
        let _ = writeln!(stdout, "No semantic configuration changes detected.");
        return Ok(());
    }

    print_section(&mut stdout, "Server", &report.server_changes);
    print_section(&mut stdout, "Backends", &report.backend_changes);
    print_section(&mut stdout, "Routes", &report.route_changes);
    print_section(&mut stdout, "Aliases", &report.alias_changes);
    print_section(&mut stdout, "Policies", &report.policy_changes);
    print_section(&mut stdout, "Tokens", &report.token_changes);
    print_section(&mut stdout, "Security & Features", &report.feature_changes);
    Ok(())
}

fn print_section(out: &mut impl Write, title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    let _ = writeln!(out, "\n{title}:");
    for item in items {
        let _ = writeln!(out, "  - {item}");
    }
}

/// Compute a semantic diff report between `current` and `candidate`.
pub fn compute_diff(current: &GatewayConfig, candidate: &GatewayConfig) -> ConfigDiffReport {
    let mut report = ConfigDiffReport::default();

    if current.server.tcp_listen != candidate.server.tcp_listen {
        report.server_changes.push(format!(
            "tcp_listen: {} -> {}",
            current.server.tcp_listen, candidate.server.tcp_listen
        ));
    }
    if current.server.unix_socket != candidate.server.unix_socket {
        report.server_changes.push(format!(
            "unix_socket: {:?} -> {:?}",
            current.server.unix_socket, candidate.server.unix_socket
        ));
    }
    if current.server.shutdown_drain_secs != candidate.server.shutdown_drain_secs {
        report.server_changes.push(format!(
            "shutdown_drain_secs: {} -> {}",
            current.server.shutdown_drain_secs, candidate.server.shutdown_drain_secs
        ));
    }
    if current.server.enable_ui != candidate.server.enable_ui {
        report.server_changes.push(format!(
            "enable_ui: {} -> {}",
            current.server.enable_ui, candidate.server.enable_ui
        ));
    }
    if current.server.enable_admin != candidate.server.enable_admin {
        report.server_changes.push(format!(
            "enable_admin: {} -> {}",
            current.server.enable_admin, candidate.server.enable_admin
        ));
    }

    let cur_backends: BTreeMap<&str, &BackendConfig> = current
        .backends
        .iter()
        .map(|b| (b.id.as_str(), b))
        .collect();
    let cand_backends: BTreeMap<&str, &BackendConfig> = candidate
        .backends
        .iter()
        .map(|b| (b.id.as_str(), b))
        .collect();
    for (id, b) in &cand_backends {
        match cur_backends.get(id) {
            None => report
                .backend_changes
                .push(format!("added backend {id:?} (kind={})", b.kind.as_str())),
            Some(old) => {
                if old.kind != b.kind {
                    report.backend_changes.push(format!(
                        "backend {id:?}: kind {} -> {}",
                        old.kind.as_str(),
                        b.kind.as_str()
                    ));
                }
                if old.timeout_secs != b.timeout_secs {
                    report.backend_changes.push(format!(
                        "backend {id:?}: timeout_secs {} -> {}",
                        old.timeout_secs, b.timeout_secs
                    ));
                }
                if old.max_concurrency != b.max_concurrency {
                    report.backend_changes.push(format!(
                        "backend {id:?}: max_concurrency {} -> {}",
                        old.max_concurrency, b.max_concurrency
                    ));
                }
                if old.command != b.command {
                    report.backend_changes.push(format!(
                        "backend {id:?}: command {:?} -> {:?}",
                        old.command, b.command
                    ));
                }
                if old.base_url != b.base_url {
                    report.backend_changes.push(format!(
                        "backend {id:?}: base_url {:?} -> {:?}",
                        old.base_url, b.base_url
                    ));
                }
            }
        }
    }
    for id in cur_backends.keys() {
        if !cand_backends.contains_key(id) {
            report
                .backend_changes
                .push(format!("removed backend {id:?}"));
        }
    }

    for (model, chain) in &candidate.routes {
        match current.routes.get(model) {
            None => report
                .route_changes
                .push(format!("added route {model:?} = {chain:?}")),
            Some(old) if old != chain => report
                .route_changes
                .push(format!("route {model:?}: {old:?} -> {chain:?}")),
            Some(_) => {}
        }
    }
    for model in current.routes.keys() {
        if !candidate.routes.contains_key(model) {
            report
                .route_changes
                .push(format!("removed route {model:?}"));
        }
    }

    for (alias, target) in &candidate.aliases {
        match current.aliases.get(alias) {
            None => report
                .alias_changes
                .push(format!("added alias {alias:?} -> {target:?}")),
            Some(old) if old != target => report
                .alias_changes
                .push(format!("alias {alias:?}: {old:?} -> {target:?}")),
            Some(_) => {}
        }
    }
    for alias in current.aliases.keys() {
        if !candidate.aliases.contains_key(alias) {
            report
                .alias_changes
                .push(format!("removed alias {alias:?}"));
        }
    }

    diff_policies(
        &current.route_policies,
        &candidate.route_policies,
        &mut report.policy_changes,
    );
    diff_tokens(
        &current.tokens,
        &candidate.tokens,
        &mut report.token_changes,
    );

    if current.security.egress_guard.mode != candidate.security.egress_guard.mode {
        report.feature_changes.push(format!(
            "security.egress_guard.mode: {:?} -> {:?}",
            current.security.egress_guard.mode, candidate.security.egress_guard.mode
        ));
    }
    if current.flight_recorder.enabled != candidate.flight_recorder.enabled {
        report.feature_changes.push(format!(
            "flight_recorder.enabled: {} -> {}",
            current.flight_recorder.enabled, candidate.flight_recorder.enabled
        ));
    }
    if current.cache.enabled != candidate.cache.enabled
        || current.cache.coalesce_enabled() != candidate.cache.coalesce_enabled()
    {
        report.feature_changes.push(format!(
            "cache: enabled {}->{}, coalesce {}->{}",
            current.cache.enabled,
            candidate.cache.enabled,
            current.cache.coalesce_enabled(),
            candidate.cache.coalesce_enabled()
        ));
    }

    report
}

fn diff_policies(
    current: &BTreeMap<String, RoutePolicyConfig>,
    candidate: &BTreeMap<String, RoutePolicyConfig>,
    out: &mut Vec<String>,
) {
    for (route, pol) in candidate {
        match current.get(route) {
            None => out.push(format!(
                "added policy {route:?} (strategy={:?}, hedge_delay_ms={})",
                pol.strategy, pol.hedge_delay_ms
            )),
            Some(old) => {
                if old.strategy != pol.strategy || old.hedge_delay_ms != pol.hedge_delay_ms {
                    out.push(format!(
                        "policy {route:?}: ({:?}, {}ms) -> ({:?}, {}ms)",
                        old.strategy, old.hedge_delay_ms, pol.strategy, pol.hedge_delay_ms
                    ));
                }
            }
        }
    }
    for route in current.keys() {
        if !candidate.contains_key(route) {
            out.push(format!("removed policy {route:?}"));
        }
    }
}

fn diff_tokens(current: &[TokenConfig], candidate: &[TokenConfig], out: &mut Vec<String>) {
    let label = |t: &TokenConfig| {
        t.name
            .clone()
            .unwrap_or_else(|| format!("hash-len-{}", t.token.len()))
    };
    let cur_map: BTreeMap<String, &TokenConfig> = current.iter().map(|t| (label(t), t)).collect();
    let cand_map: BTreeMap<String, &TokenConfig> =
        candidate.iter().map(|t| (label(t), t)).collect();
    for (name, tok) in &cand_map {
        match cur_map.get(name) {
            None => out.push(format!("added token {name:?}")),
            Some(old) => {
                if old.rpm != tok.rpm
                    || old.tpm != tok.tpm
                    || old.daily_token_budget != tok.daily_token_budget
                    || old.max_concurrent != tok.max_concurrent
                {
                    out.push(format!(
                        "token {name:?}: rpm {:?}->{:?}, tpm {:?}->{:?}, budget {:?}->{:?}, concurrent {:?}->{:?}",
                        old.rpm,
                        tok.rpm,
                        old.tpm,
                        tok.tpm,
                        old.daily_token_budget,
                        tok.daily_token_budget,
                        old.max_concurrent,
                        tok.max_concurrent
                    ));
                }
            }
        }
    }
    for name in cur_map.keys() {
        if !cand_map.contains_key(name) {
            out.push(format!("removed token {name:?}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_semantic_route_backend_and_quota_diffs() {
        let a = GatewayConfig::from_toml_str(
            r#"
[[backends]]
id = "b1"
kind = "cli"
command = "echo"
stdin = true
timeout_secs = 60

[routes]
"default" = ["b1"]
"#,
        )
        .expect("a");
        let b = GatewayConfig::from_toml_str(
            r#"
[security.egress_guard]
mode = "redact"

[[backends]]
id = "b1"
kind = "cli"
command = "echo"
stdin = true
timeout_secs = 120

[routes]
"default" = ["b1"]
"fast" = ["b1"]
"#,
        )
        .expect("b");
        let diff = compute_diff(&a, &b);
        assert!(!diff.is_empty());
        assert!(
            diff.backend_changes
                .iter()
                .any(|s| s.contains("timeout_secs 60 -> 120"))
        );
        assert!(
            diff.route_changes
                .iter()
                .any(|s| s.contains("added route \"fast\""))
        );
        assert!(diff.feature_changes.iter().any(|s| s.contains("Redact")));
    }
}
