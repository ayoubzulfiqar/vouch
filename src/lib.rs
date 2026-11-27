//! `vouch` — a local authentication multiplexer and process-supervised
//! gateway daemon for AI orchestrators.
//!
//! The crate is split into small subsystems so that the ingress adapters, the
//! routing policy, and the egress drivers can be tested and reasoned about
//! independently:
//!
//! * [`config`] — typed TOML configuration and hot-reload storage.
//! * [`error`] — the single error type used across the daemon.
//! * [`auth`] — virtual ingress tokens and local CLI/auth discovery.
//! * [`router`] — model routing, fallback chains, and session affinity.
//! * [`driver`] — HTTP, CLI, and PTY egress implementations.
//! * [`ingress`] — OpenAI, Anthropic, and operational HTTP endpoints.
//! * [`cli`] — operator-facing subcommands.
//! * [`telemetry`] — structured logging and secret redaction.
//! * [`metrics`] — Prometheus instrumentation.
//! * [`state`] — the shared, thread-safe application state.

pub mod auth;
pub mod cli;
pub mod config;
pub mod driver;
pub mod error;
pub mod ingress;
pub mod metrics;
pub mod persistence;
pub mod router;
pub mod state;
pub mod telemetry;
pub mod usage;

/// Crate version, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
