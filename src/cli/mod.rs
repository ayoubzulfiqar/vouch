//! Operator-facing command line.

pub mod bench;
pub mod doctor;
pub mod mcp;
pub mod service;
pub mod token;
pub mod top;

use std::{io::Write, net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::{config::GatewayConfig, error::Result};

/// Top-level command line.
#[derive(Debug, Parser)]
#[command(
    name = "vouch",
    version = crate::VERSION,
    about = "Local authentication multiplexer and process-supervised AI gateway"
)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the gateway in the foreground.
    Serve(ServeArgs),
    /// Print the configured model routes.
    Routes(RoutesArgs),
    /// Inspect local CLIs, credentials, daemons, and configured drivers.
    Doctor(DoctorArgs),
    /// Install or remove a user service (systemd or launchd).
    Service(ServiceArgs),
    /// Run a Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio or HTTP.
    Mcp(McpArgs),
    /// Display a live terminal dashboard polling `/healthz`, `/readyz`, and `/metrics`.
    Top(TopArgs),
    /// Generate virtual ingress tokens (`vouch token create`).
    Token(TokenArgs),
    /// Benchmark configured routes end-to-end (TTFT, latency percentiles, fallbacks).
    Bench(BenchArgs),
}

/// Arguments for `vouch serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Override `server.tcp_listen` (useful for tests and containers).
    #[arg(long)]
    pub tcp_listen: Option<SocketAddr>,
    /// Override `server.unix_socket`.
    #[arg(long)]
    pub unix_socket: Option<PathBuf>,
}

/// Arguments for `vouch routes`.
#[derive(Debug, Args)]
pub struct RoutesArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `vouch doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `vouch service`.
#[derive(Debug, Args)]
pub struct ServiceArgs {
    /// Service action.
    #[command(subcommand)]
    pub command: ServiceCommand,
}

/// Service actions.
#[derive(Debug, Subcommand)]
pub enum ServiceCommand {
    /// Install and enable the user service.
    Install(ServiceInstallArgs),
    /// Stop, disable, and remove the user service.
    Uninstall(ServiceUninstallArgs),
}

/// Arguments for `vouch service install`.
#[derive(Debug, Args)]
pub struct ServiceInstallArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Binary to run. Defaults to the current executable.
    #[arg(long)]
    pub binary: Option<PathBuf>,
    /// Write the unit but do not enable or start it.
    #[arg(long)]
    pub no_start: bool,
}

/// Arguments for `vouch service uninstall`.
#[derive(Debug, Args)]
pub struct ServiceUninstallArgs {
    /// Remove the unit but do not stop a running service.
    #[arg(long)]
    pub no_stop: bool,
}

/// Transport mode for `vouch mcp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum McpTransport {
    /// JSON-RPC 2.0 over stdin/stdout (tracing logs routed to stderr).
    Stdio,
    /// JSON-RPC 2.0 over HTTP (`POST /mcp`).
    Http,
}

/// Arguments for `vouch mcp`.
#[derive(Debug, Args)]
pub struct McpArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Transport implementation.
    #[arg(long, value_enum, default_value_t = McpTransport::Stdio)]
    pub transport: McpTransport,
    /// Bind address when `--transport http` is used.
    #[arg(long, default_value = "127.0.0.1:8081")]
    pub bind: SocketAddr,
}

/// Arguments for `vouch top`.
#[derive(Debug, Args)]
pub struct TopArgs {
    /// Base HTTP URL of the running vouch daemon.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub url: String,
    /// Connect via Unix domain socket instead of TCP.
    #[arg(long)]
    pub unix_socket: Option<PathBuf>,
    /// Refresh interval in milliseconds.
    #[arg(long, default_value_t = 1000)]
    pub interval_ms: u64,
}

/// Arguments for `vouch token`.
#[derive(Debug, Args)]
pub struct TokenArgs {
    /// Token management action.
    #[command(subcommand)]
    pub command: TokenCommand,
}

/// Token subcommands.
#[derive(Debug, Subcommand)]
pub enum TokenCommand {
    /// Generate a cryptographically random `vouch_live_...` virtual token TOML snippet.
    Create(TokenCreateArgs),
}

/// Arguments for `vouch token create`.
#[derive(Debug, Args)]
pub struct TokenCreateArgs {
    /// Optional human-readable label included in logs and metrics.
    #[arg(long)]
    pub name: Option<String>,
    /// Optional requests-per-minute sliding quota.
    #[arg(long)]
    pub rpm: Option<u32>,
    /// Allowed model glob patterns (repeat `--allow` for multiple patterns).
    #[arg(long = "allow")]
    pub allow: Vec<String>,
}

/// Arguments for `vouch bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Path to the TOML configuration.
    #[arg(short, long, default_value = "vouch.toml")]
    pub config: PathBuf,
    /// Target model route to benchmark (defaults to `defaults.model` when configured).
    #[arg(short, long)]
    pub model: Option<String>,
    /// Total number of requests to run.
    #[arg(short = 'n', long, default_value_t = 5)]
    pub requests: usize,
    /// Maximum concurrent benchmark requests.
    #[arg(long, default_value_t = 2)]
    pub concurrency: usize,
    /// Prompt text sent on each benchmark request.
    #[arg(long, default_value = "Reply with OK.")]
    pub prompt: String,
    /// Benchmark streaming mode to measure true TTFT.
    #[arg(long)]
    pub stream: bool,
    /// Emit machine-readable JSON summary.
    #[arg(long)]
    pub json: bool,
}

/// Render the configured routes as text or JSON.
pub fn print_routes(config: &GatewayConfig, json: bool) -> Result<()> {
    if json {
        let payload = serde_json::to_string_pretty(&config.routes)?;
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{payload}")?;
        return Ok(());
    }

    let mut stdout = std::io::stdout().lock();
    for (model, chain) in &config.routes {
        writeln!(stdout, "{model}")?;
        for (index, id) in chain.iter().enumerate() {
            let marker = if index == 0 { "->" } else { "  " };
            match config.backend(id) {
                Some(backend) => writeln!(
                    stdout,
                    "  {marker} {id} ({}, max_concurrency={})",
                    backend.kind.as_str(),
                    backend.max_concurrency
                )?,
                None => writeln!(stdout, "  {marker} {id} (missing)")?,
            }
        }
    }
    if config.tokens.is_empty() {
        writeln!(stdout, "\ntokens: local mode (any bearer accepted)")?;
    } else {
        writeln!(
            stdout,
            "\ntokens: {} configured, enforced",
            config.tokens.len()
        )?;
    }
    Ok(())
}
