//! Operator-facing command line.

pub mod doctor;
pub mod service;

use std::{io::Write, net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand};

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
