//! Pseudo-terminal driver for CLIs that refuse to flush output when
//! `isatty() == false` or that render a TUI before printing a result.
//!
//! `portable-pty` is blocking by design, so it runs inside
//! `tokio::task::spawn_blocking` and communicates through bounded channels.
//! The same process-group guard used by the headless driver supervises the
//! child, and dropping the returned stream kills the group.

use std::{
    io::{Read, Write},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    config::{BackendConfig, BackendKind, DefaultsConfig, OutputMode},
    driver::{
        AnsiStripper, ChunkStream, Driver, DriverResponse, ExecutionRequest, PermitGuard,
        ProcessGroupGuard, StreamChunk, emit_subprocess_tail, finalize_subprocess_completion,
        io_error, strip_ansi,
    },
    error::{GatewayError, Result},
    metrics,
};

const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const PTY_ROWS: u16 = 40;
const PTY_COLS: u16 = 120;

#[derive(Debug)]
struct PtyExit {
    success: bool,
    code: i32,
    signal: Option<String>,
}

impl PtyExit {
    fn label(&self) -> String {
        match self.signal.as_deref() {
            Some(signal) => format!("signal {signal}"),
            None => format!("exit code {}", self.code),
        }
    }

    fn metric_code(&self) -> String {
        self.signal.clone().unwrap_or_else(|| self.code.to_string())
    }
}

/// PTY-backed driver.
#[derive(Debug)]
pub struct PtyRunnerDriver {
    config: BackendConfig,
    semaphore: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl PtyRunnerDriver {
    /// Build a driver from backend configuration.
    pub fn new(config: BackendConfig, defaults: &DefaultsConfig) -> Result<Self> {
        let queue_timeout = Duration::from_millis(
            config
                .queue_timeout_ms
                .unwrap_or(defaults.queue_timeout_ms)
                .max(1),
        );
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrency.max(1))),
            config,
            queue_timeout,
        })
    }

    /// Spawn the PTY session and return the raw byte channel plus lifecycle
    /// handles.
    async fn spawn_session(
        &self,
        request: &ExecutionRequest,
    ) -> Result<(
        mpsc::Receiver<Result<Vec<u8>>>,
        oneshot::Receiver<PtyExit>,
        ProcessGroupGuard,
    )> {
        let command = self.config.command.clone().ok_or_else(|| {
            GatewayError::Config(format!("backend {} has no command", self.config.id))
        })?;
        let prompt = request.rendered_prompt();
        let args: Vec<String> = self
            .config
            .args
            .iter()
            .map(|arg| {
                arg.replace("{prompt}", &prompt)
                    .replace("{model}", &request.model)
                    .replace("{session_id}", &request.session_key)
                    .replace("{request_id}", &request.request_id)
            })
            .collect();
        let env = self.config.env.clone();
        let workdir = request
            .workdir
            .clone()
            .or_else(|| self.config.workdir.clone());
        let stdin = self.config.stdin;
        let send_eof = self.config.pty_send_eof;
        let backend = self.config.id.clone();
        let request_id = request.request_id.clone();
        let session_key = request.session_key.clone();
        let model = request.model.clone();

        let (data_sender, data_receiver) = mpsc::channel::<Result<Vec<u8>>>(32);
        let (exit_sender, exit_receiver) = oneshot::channel::<PtyExit>();
        let (setup_sender, setup_receiver) =
            oneshot::channel::<std::result::Result<ProcessGroupGuard, String>>();

        tokio::task::spawn_blocking(move || {
            let pty_system = native_pty_system();
            let pair = match pty_system.openpty(PtySize {
                rows: PTY_ROWS,
                cols: PTY_COLS,
                pixel_width: 0,
                pixel_height: 0,
            }) {
                Ok(pair) => pair,
                Err(error) => {
                    let _ = setup_sender.send(Err(format!("cannot open pty: {error}")));
                    return;
                }
            };

            let mut builder = CommandBuilder::new(command);
            builder.args(args);
            for (key, value) in env {
                builder.env(key, value);
            }
            builder.env("VOUCH_REQUEST_ID", &request_id);
            builder.env("VOUCH_SESSION_ID", &session_key);
            builder.env("VOUCH_MODEL", &model);
            if let Some(workdir) = workdir {
                builder.cwd(workdir);
            }

            let mut child = match pair.slave.spawn_command(builder) {
                Ok(child) => child,
                Err(error) => {
                    let _ = setup_sender.send(Err(format!("cannot spawn pty child: {error}")));
                    return;
                }
            };
            drop(pair.slave);

            let Some(pid) = child.process_id() else {
                let _ = setup_sender.send(Err("pty child did not report a pid".into()));
                let _ = child.kill();
                return;
            };
            let guard = ProcessGroupGuard::new(pid, backend.clone());
            if setup_sender.send(Ok(guard.clone_guard())).is_err() {
                guard.terminate();
                let _ = child.kill();
                return;
            }

            let mut reader = match pair.master.try_clone_reader() {
                Ok(reader) => reader,
                Err(error) => {
                    let _ = data_sender.blocking_send(Err(GatewayError::Internal(format!(
                        "cannot read pty: {error}"
                    ))));
                    guard.terminate();
                    let _ = child.wait();
                    return;
                }
            };
            if stdin {
                match pair.master.take_writer() {
                    Ok(mut writer) => {
                        let _ = writer.write_all(prompt.as_bytes());
                        if !prompt.ends_with('\n') {
                            let _ = writer.write_all(b"\n");
                        }
                        if send_eof {
                            // EOT in canonical mode asks the terminal line
                            // discipline for EOF, which many REPLs honor.
                            let _ = writer.write_all(b"\x04");
                        }
                        let _ = writer.flush();
                        // Dropping the writer also sends EOF.
                    }
                    Err(error) => {
                        let _ = data_sender.blocking_send(Err(GatewayError::Internal(format!(
                            "cannot write pty: {error}"
                        ))));
                    }
                }
            }

            let mut buffer = [0u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => {
                        if data_sender
                            .blocking_send(Ok(buffer[..read].to_vec()))
                            .is_err()
                        {
                            guard.terminate();
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = data_sender
                            .blocking_send(Err(io_error("cannot read pty output", error)));
                        guard.terminate();
                        break;
                    }
                }
            }

            match child.wait() {
                Ok(status) => {
                    if status.success() {
                        guard.disarm();
                    }
                    let _ = exit_sender.send(PtyExit {
                        success: status.success(),
                        code: status.exit_code() as i32,
                        signal: status.signal().map(ToString::to_string),
                    });
                }
                Err(error) => {
                    guard.terminate();
                    let _ = exit_sender.send(PtyExit {
                        success: false,
                        code: -1,
                        signal: Some(format!("wait failed: {error}")),
                    });
                }
            }
        });

        let guard = tokio::time::timeout(Duration::from_secs(10), setup_receiver)
            .await
            .map_err(|_| GatewayError::Upstream("pty setup timed out".into()))?
            .map_err(|_| GatewayError::Upstream("pty setup task failed".into()))?
            .map_err(GatewayError::Upstream)?;
        Ok((data_receiver, exit_receiver, guard))
    }

    async fn complete(
        &self,
        request: &ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let _permit = permit;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let (mut data, exit, guard) = self.spawn_session(request).await?;
        let prompt = request.rendered_prompt();

        let collect = async {
            let mut raw = Vec::new();
            while let Some(item) = data.recv().await {
                let bytes = item?;
                if raw.len() + bytes.len() > MAX_OUTPUT_BYTES {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exceeded the {} byte output limit",
                        self.config.id, MAX_OUTPUT_BYTES
                    )));
                }
                raw.extend_from_slice(&bytes);
            }
            let exit = exit
                .await
                .map_err(|_| GatewayError::Upstream("pty exit channel closed".into()))?;
            Ok::<_, GatewayError>((raw, exit))
        };

        let outcome = tokio::select! {
            _ = request.cancel.cancelled() => Err(GatewayError::ClientDisconnected),
            result = tokio::time::timeout(timeout, collect) => match result {
                Ok(result) => result,
                Err(_) => Err(GatewayError::Timeout(timeout)),
            },
        };

        match outcome {
            Ok((raw, exit)) => {
                guard.disarm();
                metrics::record_child_exit(&self.config.id, &exit.metric_code());
                if !exit.success {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exited with {}",
                        self.config.id,
                        exit.label()
                    )));
                }
                let text = strip_ansi(&raw);
                Ok(DriverResponse::Complete(finalize_subprocess_completion(
                    &request.model,
                    &prompt,
                    &text,
                    self.config.output_mode,
                    !request.tools.is_empty(),
                )?))
            }
            Err(error) => {
                guard.terminate();
                metrics::record_child_exit(&self.config.id, "killed");
                Err(error)
            }
        }
    }

    async fn stream(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let (mut data, exit, guard) = self.spawn_session(&request).await?;
        let backend = self.config.id.clone();
        let output_mode = self.config.output_mode;
        let parse_tools = !request.tools.is_empty();
        let model = request.model.clone();
        let cancel = request.cancel.clone();
        let prompt = request.rendered_prompt();
        let (sender, receiver) = mpsc::channel::<Result<StreamChunk>>(32);

        tokio::spawn(async move {
            let _permit = permit;
            let deadline = tokio::time::Instant::now() + timeout;
            let mut stripper = AnsiStripper::new();
            let mut json_buffer = String::new();
            let mut line_buffer = String::new();
            let mut produced = String::new();

            loop {
                let item = tokio::select! {
                    _ = cancel.cancelled() => {
                        guard.terminate();
                        return;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        guard.terminate();
                        metrics::record_child_exit(&backend, "killed");
                        let _ = sender.send(Err(GatewayError::Timeout(timeout))).await;
                        return;
                    }
                    item = data.recv() => item,
                };
                let Some(item) = item else {
                    break;
                };
                let bytes = match item {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        guard.terminate();
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                };
                let stripped = stripper.feed(&bytes);
                if stripped.is_empty() {
                    continue;
                }
                match output_mode {
                    OutputMode::Text => {
                        if produced.len() + stripped.len() > MAX_OUTPUT_BYTES {
                            guard.terminate();
                            let _ = sender
                                .send(Err(GatewayError::Upstream(format!(
                                    "backend {backend} exceeded the {MAX_OUTPUT_BYTES} byte output limit"
                                ))))
                                .await;
                            return;
                        }
                        produced.push_str(&stripped);
                        if !parse_tools
                            && sender.send(Ok(StreamChunk::delta(stripped))).await.is_err()
                        {
                            guard.terminate();
                            return;
                        }
                    }
                    OutputMode::Json => json_buffer.push_str(&stripped),
                    OutputMode::JsonLines => {
                        line_buffer.push_str(&stripped);
                        while let Some(position) = line_buffer.find('\n') {
                            let line: String = line_buffer.drain(..=position).collect();
                            let line = line.trim_end_matches(['\n', '\r']);
                            if line.trim().is_empty() {
                                continue;
                            }
                            let delta = crate::driver::cli_pipe::jsonl_line_to_delta(line);
                            produced.push_str(&delta);
                            if !delta.is_empty()
                                && sender.send(Ok(StreamChunk::delta(delta))).await.is_err()
                            {
                                guard.terminate();
                                return;
                            }
                        }
                    }
                }
            }

            let exit = exit
                .await
                .map_err(|_| GatewayError::Upstream("pty exit channel closed".into()));
            let exit = match exit {
                Ok(exit) => exit,
                Err(error) => {
                    guard.terminate();
                    let _ = sender.send(Err(error)).await;
                    return;
                }
            };

            if output_mode == OutputMode::Json {
                match crate::driver::cli_pipe::parse_json_content(&json_buffer) {
                    Ok(content) if !content.is_empty() => {
                        produced.push_str(&content);
                        if sender.send(Ok(StreamChunk::delta(content))).await.is_err() {
                            guard.terminate();
                            return;
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        guard.terminate();
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                }
            } else if output_mode == OutputMode::JsonLines && !line_buffer.trim().is_empty() {
                let delta = crate::driver::cli_pipe::jsonl_line_to_delta(line_buffer.trim());
                if !delta.is_empty() {
                    produced.push_str(&delta);
                    if sender.send(Ok(StreamChunk::delta(delta))).await.is_err() {
                        guard.terminate();
                        return;
                    }
                }
            }

            metrics::record_child_exit(&backend, &exit.metric_code());
            if exit.success {
                guard.disarm();
                let _ = emit_subprocess_tail(
                    &sender,
                    &model,
                    &prompt,
                    &produced,
                    output_mode,
                    parse_tools,
                )
                .await;
            } else {
                guard.disarm();
                let _ = sender
                    .send(Err(GatewayError::Upstream(format!(
                        "backend {backend} exited with {}",
                        exit.label()
                    ))))
                    .await;
            }
        });

        let stream: ChunkStream = Box::pin(ReceiverStream::new(receiver));
        Ok(DriverResponse::Stream(stream))
    }
}

#[async_trait]
impl Driver for PtyRunnerDriver {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Pty
    }

    fn max_concurrency(&self) -> usize {
        self.config.max_concurrency
    }

    async fn execute(&self, request: ExecutionRequest) -> Result<DriverResponse> {
        let permit = PermitGuard::acquire(
            self.semaphore.clone(),
            &self.config.id,
            self.queue_timeout,
            &request.cancel,
        )
        .await?;
        if request.stream {
            self.stream(request, permit).await
        } else {
            self.complete(&request, permit).await
        }
    }
}
