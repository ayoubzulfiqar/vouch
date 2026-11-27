//! `vouch` daemon entrypoint.
//!
//! Parses the CLI, installs redacting structured logging, builds the shared
//! application state, and serves the OpenAI/Anthropic adapters over TCP and
//! (optionally) a Unix domain socket. `SIGINT`/`SIGTERM` trigger graceful
//! shutdown; `SIGHUP` and configuration-file changes trigger an atomic hot
//! reload that never drops in-flight requests.

use std::{path::Path, sync::Arc, time::Duration};

use clap::Parser;
use notify::Watcher as _;
use tokio::{
    net::{TcpListener, UnixListener},
    task::JoinSet,
};
use tracing::{error, info, warn};

use vouch::{
    cli::{Cli, Command, ServeArgs, ServiceCommand, doctor, mcp, print_routes, service, top},
    config::GatewayConfig,
    error::{GatewayError, Result},
    ingress::build_router,
    state::AppState,
    telemetry,
};

#[tokio::main]
async fn main() {
    if let Err(error) = real_main().await {
        eprintln!("vouch: {error}");
        std::process::exit(1);
    }
}

async fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve(args).await,
        Command::Routes(args) => {
            let config = GatewayConfig::load(&args.config)?;
            print_routes(&config, args.json)
        }
        Command::Doctor(args) => {
            let healthy = doctor::run(&args).await?;
            if !healthy {
                std::process::exit(2);
            }
            Ok(())
        }
        Command::Service(args) => match args.command {
            ServiceCommand::Install(_) | ServiceCommand::Uninstall(_) => service::run(&args),
        },
        Command::Mcp(args) => mcp::run(&args).await,
        Command::Top(args) => top::run(&args).await,
    }
}

/// Run the gateway until a termination signal arrives.
async fn serve(args: ServeArgs) -> Result<()> {
    let mut config = GatewayConfig::load(&args.config)?;
    if let Some(listen) = args.tcp_listen {
        config.server.tcp_listen = listen;
    }
    if let Some(socket) = args.unix_socket {
        config.server.unix_socket = Some(socket);
    }

    let log_level = config.server.log_level.clone();
    let secrets = config.secrets();
    telemetry::init(&log_level, secrets)?;

    let state = AppState::new(args.config, config)?;
    state.start_background_tasks();
    let app = build_router(state.clone());
    let shutdown = state.shutdown_token();

    let mut servers = JoinSet::new();
    let tcp = TcpListener::bind(state.config().server.tcp_listen)
        .await
        .map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot bind {}: {error}", state.config().server.tcp_listen),
            ))
        })?;
    let tcp_addr = tcp.local_addr().map_err(GatewayError::Io)?;
    info!(address = %tcp_addr, "TCP listener ready");
    let tcp_shutdown = shutdown.clone();
    let tcp_app = app.clone();
    servers.spawn(async move {
        if let Err(error) = axum::serve(tcp, tcp_app)
            .with_graceful_shutdown(async move { tcp_shutdown.cancelled().await })
            .await
        {
            error!(error = %error, "TCP server failed");
        }
    });

    let socket_path = state.config().server.unix_socket.clone();
    if let Some(path) = &socket_path {
        prepare_socket_path(path)?;
        let listener = UnixListener::bind(path).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot bind Unix socket {}: {error}", path.display()),
            ))
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(
                |error| {
                    GatewayError::Io(std::io::Error::new(
                        error.kind(),
                        format!("cannot chmod {}: {error}", path.display()),
                    ))
                },
            )?;
        }
        info!(path = %path.display(), "Unix socket listener ready");
        let uds_shutdown = shutdown.clone();
        let uds_app = app.clone();
        servers.spawn(async move {
            if let Err(error) = axum::serve(listener, uds_app)
                .with_graceful_shutdown(async move { uds_shutdown.cancelled().await })
                .await
            {
                error!(error = %error, "Unix socket server failed");
            }
        });
    }

    let reload = start_reload_task(state.clone());

    wait_for_termination().await?;
    info!("shutdown requested; draining in-flight requests");
    state.shutdown_token().cancel();
    reload.abort();
    while servers.join_next().await.is_some() {}
    state.shutdown().await;

    if let Some(path) = &socket_path {
        let _ = std::fs::remove_file(path);
    }
    info!("vouch stopped");
    Ok(())
}

/// Remove a stale socket and create its parent directory if needed.
fn prepare_socket_path(path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_file(path).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot remove stale socket {}: {error}", path.display()),
            ))
        })?;
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot create {}: {error}", parent.display()),
            ))
        })?;
    }
    Ok(())
}

/// Wait for `SIGINT` or `SIGTERM` (or `ctrl_c` on non-Unix hosts).
#[cfg(unix)]
async fn wait_for_termination() -> Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot listen for SIGINT: {error}"),
        ))
    })?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot listen for SIGTERM: {error}"),
        ))
    })?;
    tokio::select! {
        _ = interrupt.recv() => {}
        _ = terminate.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
async fn wait_for_termination() -> Result<()> {
    tokio::signal::ctrl_c()
        .await
        .map_err(|error| GatewayError::Io(error))
}

/// Spawn the hot-reload supervisor.
fn start_reload_task(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        reload_loop(state).await;
    })
}

/// Watch `SIGHUP` and the configuration file for changes.
#[cfg(unix)]
async fn reload_loop(state: Arc<AppState>) {
    use tokio::signal::unix::{SignalKind, signal};

    let mut hangup = match signal(SignalKind::hangup()) {
        Ok(stream) => stream,
        Err(error) => {
            warn!(error = %error, "cannot listen for SIGHUP; file watching only");
            return file_watch_loop(state).await;
        }
    };

    let file_task = spawn_file_watcher(state.clone());
    let shutdown = state.shutdown_token();
    loop {
        tokio::select! {
            _ = hangup.recv() => {
                info!("SIGHUP received; reloading configuration");
                if let Err(error) = state.reload().await {
                    warn!(error = %error, "configuration reload failed; keeping previous snapshot");
                }
            }
            _ = shutdown.cancelled() => break,
        }
    }
    file_task.abort();
}

#[cfg(not(unix))]
async fn reload_loop(state: Arc<AppState>) {
    let file_task = spawn_file_watcher(state.clone());
    state.shutdown_token().cancelled().await;
    file_task.abort();
}

/// Watch the configuration file's parent directory using `notify`.
fn spawn_file_watcher(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    let (sender, receiver) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
    let config_path = state.config_path().to_path_buf();
    let file_name = config_path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();

    let mut watcher = match notify::recommended_watcher(sender) {
        Ok(watcher) => watcher,
        Err(error) => {
            warn!(error = %error, "cannot start configuration file watcher");
            return tokio::spawn(async {});
        }
    };
    let target = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if let Err(error) = watcher.watch(target, notify::RecursiveMode::NonRecursive) {
        warn!(error = %error, path = %target.display(), "cannot watch configuration directory");
        return tokio::spawn(async {});
    }

    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        // Keep the watcher alive for as long as this task runs.
        let _watcher = watcher;
        loop {
            if state.shutdown_token().is_cancelled() {
                break;
            }
            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok(Ok(event)) => {
                    let relevant =
                        event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove();
                    let matches = event.paths.iter().any(|path| {
                        path.file_name()
                            .map(|name| name == file_name)
                            .unwrap_or(false)
                    });
                    if relevant && matches {
                        while matches!(receiver.try_recv(), Ok(Ok(_))) {}
                        std::thread::sleep(Duration::from_millis(75));
                        handle.block_on(async {
                            if let Err(error) = state.reload().await {
                                warn!(error = %error, "configuration file reload failed; keeping previous snapshot");
                            }
                        });
                    }
                }
                Ok(Err(error)) => warn!(error = %error, "file watcher event error"),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    })
}

#[cfg(unix)]
async fn file_watch_loop(state: Arc<AppState>) {
    let task = spawn_file_watcher(state.clone());
    state.shutdown_token().cancelled().await;
    task.abort();
}
