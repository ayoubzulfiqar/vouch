//! Headless stdin/stdout subprocess driver.
//!
//! The child is always placed in its own process group. A
//! [`ProcessGroupGuard`] is moved into the task that owns the child, so a
//! timeout, I/O error, or dropped SSE receiver tears down the whole group
//! (including grandchildren) instead of leaking processes.

use std::{process::Stdio, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{Semaphore, mpsc},
};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    config::{BackendConfig, BackendKind, DefaultsConfig, OutputMode},
    driver::{
        AnsiStripper, ChunkStream, Driver, DriverResponse, ExecutionRequest, PermitGuard,
        ProcessGroupGuard, StreamChunk, emit_subprocess_tail, extract_content,
        finalize_subprocess_completion, io_error, strip_ansi, truncate,
    },
    error::{GatewayError, Result},
    metrics, telemetry,
};

/// Maximum buffered stdout for non-streaming executions.
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum captured stderr used in diagnostics.
const MAX_STDERR_BYTES: usize = 8 * 1024;

/// Headless CLI pipe driver.
#[derive(Debug)]
pub struct CliPipeDriver {
    config: BackendConfig,
    semaphore: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl CliPipeDriver {
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

    fn spawn(&self, request: &ExecutionRequest) -> Result<(Child, ProcessGroupGuard)> {
        let command = self.config.command.as_deref().ok_or_else(|| {
            GatewayError::Config(format!("backend {} has no command", self.config.id))
        })?;
        let prompt = request.rendered_prompt();
        let mut command = Command::new(command);
        for arg in &self.config.args {
            command.arg(render_arg(arg, &prompt, request));
        }
        command.envs(&self.config.env);
        command.env("VOUCH_REQUEST_ID", &request.request_id);
        command.env("VOUCH_SESSION_ID", &request.session_key);
        command.env("VOUCH_MODEL", &request.model);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        if let Some(workdir) = request.workdir.as_ref().or(self.config.workdir.as_ref()) {
            command.current_dir(workdir);
        }
        let child = command.spawn().map_err(|error| {
            GatewayError::Upstream(format!(
                "cannot spawn backend {} ({}): {error}",
                self.config.id,
                command_name(&self.config)
            ))
        })?;
        let pid = child.id().ok_or_else(|| {
            GatewayError::Internal(format!("backend {} did not report a pid", self.config.id))
        })?;
        let guard = ProcessGroupGuard::new(pid, self.config.id.clone());
        Ok((child, guard))
    }

    fn write_stdin(&self, child: &mut Child, prompt: String) -> tokio::task::JoinHandle<()> {
        let stdin = child.stdin.take();
        if !self.config.stdin {
            // The pipe exists so the child sees a pipe, but it is closed
            // immediately to signal EOF.
            drop(stdin);
            return tokio::spawn(async {});
        }
        tokio::spawn(async move {
            let Some(mut stdin) = stdin else {
                return;
            };
            if stdin.write_all(prompt.as_bytes()).await.is_err() {
                return;
            }
            if !prompt.ends_with('\n') && stdin.write_all(b"\n").await.is_err() {
                return;
            }
            let _ = stdin.shutdown().await;
        })
    }

    fn drain_stderr(&self, child: &mut Child) -> tokio::task::JoinHandle<String> {
        let stderr = child.stderr.take();
        tokio::spawn(async move {
            let Some(mut stderr) = stderr else {
                return String::new();
            };
            let mut captured = Vec::new();
            let mut buffer = [0u8; 4096];
            loop {
                match stderr.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(read) => {
                        if captured.len() < MAX_STDERR_BYTES {
                            let remaining = MAX_STDERR_BYTES - captured.len();
                            captured.extend_from_slice(&buffer[..read.min(remaining)]);
                        }
                    }
                    Err(_) => break,
                }
            }
            String::from_utf8_lossy(&captured).into_owned()
        })
    }

    async fn execute_complete(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let _permit = permit;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let (mut child, guard) = self.spawn(&request)?;
        let prompt = request.rendered_prompt();
        let stdin_task = self.write_stdin(&mut child, prompt.clone());
        let stderr_task = self.drain_stderr(&mut child);

        let outcome = tokio::time::timeout(timeout, async {
            let mut stdout = child.stdout.take().ok_or_else(|| {
                GatewayError::Internal(format!("backend {} has no stdout", self.config.id))
            })?;
            let mut buffer = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let read = stdout.read(&mut chunk).await?;
                if read == 0 {
                    break;
                }
                if buffer.len() + read > MAX_OUTPUT_BYTES {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exceeded the {} byte output limit",
                        self.config.id, MAX_OUTPUT_BYTES
                    )));
                }
                buffer.extend_from_slice(&chunk[..read]);
            }
            let status = child.wait().await?;
            Ok::<_, GatewayError>((buffer, status))
        })
        .await;

        match outcome {
            Ok(Ok((buffer, status))) => {
                guard.disarm();
                let _ = tokio::time::timeout(Duration::from_secs(1), stdin_task).await;
                let stderr = tokio::time::timeout(Duration::from_secs(1), stderr_task)
                    .await
                    .ok()
                    .and_then(std::result::Result::ok)
                    .unwrap_or_default();
                record_exit(&self.config.id, &status);
                if !status.success() {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exited with {}: {}",
                        self.config.id,
                        status_label(&status),
                        truncate(&telemetry::redact_text(stderr.trim()), 500)
                    )));
                }
                let raw = strip_ansi(&buffer);
                Ok(DriverResponse::Complete(finalize_subprocess_completion(
                    &request.model,
                    &prompt,
                    &raw,
                    self.config.output_mode,
                    !request.tools.is_empty(),
                )?))
            }
            Ok(Err(error)) => {
                guard.terminate();
                let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
                let _ = stdin_task.await;
                metrics::record_child_exit(&self.config.id, "error");
                Err(error)
            }
            Err(_) => {
                guard.terminate();
                let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
                let _ = stdin_task.await;
                metrics::record_child_exit(&self.config.id, "killed");
                Err(GatewayError::Timeout(timeout))
            }
        }
    }

    async fn execute_stream(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let (mut child, guard) = self.spawn(&request)?;
        let prompt = request.rendered_prompt();
        let stdin_task = self.write_stdin(&mut child, prompt.clone());
        let stderr_task = self.drain_stderr(&mut child);
        let backend = self.config.id.clone();
        let output_mode = self.config.output_mode;
        let parse_tools = !request.tools.is_empty();
        let model = request.model.clone();
        let cancel = request.cancel.clone();
        let (sender, receiver) = mpsc::channel::<Result<StreamChunk>>(32);

        tokio::spawn(async move {
            let _permit = permit;
            let mut stdout = match child.stdout.take() {
                Some(stdout) => stdout,
                None => {
                    let _ = sender
                        .send(Err(GatewayError::Internal(format!(
                            "backend {backend} has no stdout"
                        ))))
                        .await;
                    guard.terminate();
                    let _ = child.wait().await;
                    return;
                }
            };
            let deadline = tokio::time::Instant::now() + timeout;
            let mut stripper = AnsiStripper::new();
            let mut json_buffer = String::new();
            let mut line_buffer = String::new();
            let mut produced = String::new();
            let mut buffer = [0u8; 8192];

            loop {
                let read = tokio::select! {
                    _ = cancel.cancelled() => {
                        guard.terminate();
                        let _ = child.wait().await;
                        return;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        guard.terminate();
                        let _ = child.wait().await;
                        metrics::record_child_exit(&backend, "killed");
                        let _ = sender.send(Err(GatewayError::Timeout(timeout))).await;
                        return;
                    }
                    result = stdout.read(&mut buffer) => result,
                };
                match read {
                    Ok(0) => break,
                    Ok(read) => {
                        let stripped = stripper.feed(&buffer[..read]);
                        if stripped.is_empty() {
                            continue;
                        }
                        if produced.len() + stripped.len() > MAX_OUTPUT_BYTES {
                            guard.terminate();
                            let _ = child.wait().await;
                            let _ = sender
                                .send(Err(GatewayError::Upstream(format!(
                                    "backend {backend} exceeded the {MAX_OUTPUT_BYTES} byte output limit"
                                ))))
                                .await;
                            return;
                        }
                        produced.push_str(&stripped);
                        match output_mode {
                            OutputMode::Text => {
                                if !parse_tools
                                    && sender.send(Ok(StreamChunk::delta(stripped))).await.is_err()
                                {
                                    guard.terminate();
                                    let _ = child.wait().await;
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
                                    let delta = jsonl_line_to_delta(line);
                                    if !delta.is_empty()
                                        && sender.send(Ok(StreamChunk::delta(delta))).await.is_err()
                                    {
                                        guard.terminate();
                                        let _ = child.wait().await;
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => {
                        guard.terminate();
                        let _ = child.wait().await;
                        metrics::record_child_exit(&backend, "error");
                        let _ = sender
                            .send(Err(io_error("cannot read backend stdout", error)))
                            .await;
                        return;
                    }
                }
            }

            let status = child.wait().await;
            let stderr = stderr_task.await.unwrap_or_default();
            let _ = tokio::time::timeout(Duration::from_secs(1), stdin_task).await;

            if output_mode == OutputMode::Json {
                match parse_json_content(&json_buffer) {
                    Ok(content) if !content.is_empty() => {
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
                let delta = jsonl_line_to_delta(line_buffer.trim());
                if !delta.is_empty() && sender.send(Ok(StreamChunk::delta(delta))).await.is_err() {
                    guard.terminate();
                    return;
                }
            }

            match status {
                Ok(status) if status.success() => {
                    guard.disarm();
                    record_exit(&backend, &status);
                    let _ = emit_subprocess_tail(
                        &sender,
                        &model,
                        &prompt,
                        &produced,
                        output_mode,
                        parse_tools,
                    )
                    .await;
                }
                Ok(status) => {
                    record_exit(&backend, &status);
                    guard.disarm();
                    let _ = sender
                        .send(Err(GatewayError::Upstream(format!(
                            "backend {backend} exited with {}: {}",
                            status_label(&status),
                            truncate(&telemetry::redact_text(stderr.trim()), 500)
                        ))))
                        .await;
                }
                Err(error) => {
                    let _ = sender
                        .send(Err(io_error("cannot wait for backend", error)))
                        .await;
                }
            }
        });

        let stream: ChunkStream = Box::pin(ReceiverStream::new(receiver));
        Ok(DriverResponse::Stream(stream))
    }
}

#[async_trait]
impl Driver for CliPipeDriver {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cli
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
            self.execute_stream(request, permit).await
        } else {
            self.execute_complete(request, permit).await
        }
    }
}

fn render_arg(template: &str, prompt: &str, request: &ExecutionRequest) -> String {
    template
        .replace("{prompt}", prompt)
        .replace("{model}", &request.model)
        .replace("{session_id}", &request.session_key)
        .replace("{request_id}", &request.request_id)
}

fn command_name(config: &BackendConfig) -> &str {
    config.command.as_deref().unwrap_or("<none>")
}

fn record_exit(backend: &str, status: &std::process::ExitStatus) {
    let code = status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    metrics::record_child_exit(backend, &code);
}

fn status_label(status: &std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exit code {code}"),
        None => "a signal".to_string(),
    }
}

pub(crate) fn parse_json_content(raw: &str) -> Result<String> {
    let value: Value = serde_json::from_str(raw.trim()).map_err(|error| {
        GatewayError::Upstream(format!("backend emitted invalid JSON: {error}"))
    })?;
    if let Some(error) = crate::driver::extract_upstream_error(&value) {
        return Err(GatewayError::Upstream(format!(
            "backend reported an error: {}",
            truncate(&telemetry::redact_text(&error), 500)
        )));
    }
    extract_content(&value).ok_or_else(|| {
        GatewayError::Upstream("backend JSON did not contain assistant content".into())
    })
}

pub(crate) fn parse_json_lines(raw: &str) -> String {
    let mut output = String::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        output.push_str(&jsonl_line_to_delta(line));
    }
    output
}

pub(crate) fn jsonl_line_to_delta(line: &str) -> String {
    match serde_json::from_str::<Value>(line.trim()) {
        Ok(value) => {
            let (delta, _, _) = crate::driver::extract_delta(&value);
            delta
                .or_else(|| extract_content(&value))
                .unwrap_or_default()
        }
        Err(_) => line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_argument_templates() {
        let request = ExecutionRequest {
            request_id: "req".into(),
            session_key: "sess".into(),
            model: "coding/default".into(),
            system: None,
            messages: Vec::new(),
            prompt: "hello".into(),
            stream: false,
            temperature: None,
            max_tokens: None,
            workspace_mode: crate::driver::WorkspaceMode::Auto,
            workdir: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            tools: Vec::new(),
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(render_arg("--p={prompt}", "hello", &request), "--p=hello");
        assert_eq!(render_arg("{model}", "hello", &request), "coding/default");
    }

    #[test]
    fn parses_json_lines() {
        let raw = "{\"delta\":\"a\"}\n{\"delta\":\"b\"}\n";
        assert_eq!(parse_json_lines(raw), "ab");
    }

    #[test]
    fn rejects_invalid_json_output() {
        assert!(parse_json_content("not json").is_err());
    }
}
