//! SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in redacted JSONL flight recorder (`[flight_recorder]`) for offline replay
//! and regression benchmarking (`vouch replay`).

use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use arc_swap::ArcSwap;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::{
    config::FlightRecorderConfig,
    driver::{ChatMessage, ContentPart, MessageContent},
    error::{GatewayError, Result},
    telemetry::redact_text,
};

const RECORDER_QUEUE_CAPACITY: usize = 256;

/// One sanitized request/outcome entry appended to the flight recorder JSONL log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightRecord {
    /// RFC-3339 UTC timestamp.
    pub timestamp: String,
    /// Wire protocol (`/v1/chat/completions`, `/v1/messages`, ...).
    pub protocol: String,
    /// Public model or alias requested by the caller.
    pub requested_model: String,
    /// Resolved canonical `[routes]` target.
    pub resolved_route: String,
    /// Backend that handled or completed the call.
    pub winning_backend: String,
    /// Credential profile selected, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    /// True when failover or hedged backup succeeded.
    pub fallback_used: bool,
    /// Cache/singleflight outcome (`hit`, `miss`, `bypass`, `disabled`, `coalesced`).
    pub cache_outcome: String,
    /// Time to first chunk / response completion in milliseconds.
    pub ttft_ms: u64,
    /// Total end-to-end request duration in milliseconds.
    pub total_ms: u64,
    /// Final HTTP status code.
    pub status: u16,
    /// Prompt BPE tokens recorded.
    pub prompt_tokens: u64,
    /// Completion BPE tokens recorded.
    pub completion_tokens: u64,
    /// System prompt (redacted), when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Normalized non-system conversation turns (redacted, base64 stripped).
    pub normalized_messages: Vec<ChatMessage>,
    /// Normalized tool definitions supplied in the request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// OpenAI-style `response_format` constraint when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

enum RecorderMsg {
    Record(Box<FlightRecord>),
    Flush(oneshot::Sender<()>),
}

/// Non-blocking bounded asynchronous JSONL flight recorder.
#[derive(Debug)]
pub struct FlightRecorder {
    config: ArcSwap<FlightRecorderConfig>,
    state_dir: ArcSwap<Option<PathBuf>>,
    tx: mpsc::Sender<RecorderMsg>,
    rx: ParkingMutex<Option<mpsc::Receiver<RecorderMsg>>>,
    worker_started: AtomicBool,
}

impl FlightRecorder {
    /// Build a recorder from `[flight_recorder]` and `defaults.state_dir`.
    pub fn new(config: FlightRecorderConfig, state_dir: Option<PathBuf>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(RECORDER_QUEUE_CAPACITY);
        Arc::new(Self {
            config: ArcSwap::from_pointee(config),
            state_dir: ArcSwap::from_pointee(state_dir),
            tx,
            rx: ParkingMutex::new(Some(rx)),
            worker_started: AtomicBool::new(false),
        })
    }

    /// Update configuration atomically during hot reload.
    pub fn update_config(&self, config: FlightRecorderConfig, state_dir: Option<PathBuf>) {
        self.config.store(Arc::new(config));
        self.state_dir.store(Arc::new(state_dir));
    }

    /// True when `[flight_recorder].enabled` is turned on.
    pub fn is_enabled(&self) -> bool {
        self.config.load().enabled
    }

    /// Effective JSONL destination path.
    pub fn resolved_path(&self) -> PathBuf {
        let cfg = self.config.load();
        let state_dir = self.state_dir.load();
        resolve_flight_path(&cfg, state_dir.as_deref())
    }

    /// Enqueue a sanitized flight record without blocking ingress handlers.
    pub fn record(&self, mut entry: FlightRecord) {
        if !self.is_enabled() {
            return;
        }
        sanitize_record(&mut entry);
        if self
            .tx
            .try_send(RecorderMsg::Record(Box::new(entry)))
            .is_err()
        {
            tracing::debug!("flight recorder queue full; dropping entry to preserve latency");
        }
    }

    /// Flush queued flight records to disk (used during shutdown and tests).
    pub async fn flush(&self) {
        if !self.worker_started.load(Ordering::SeqCst) {
            return;
        }
        let (done_tx, done_rx) = oneshot::channel();
        if self.tx.send(RecorderMsg::Flush(done_tx)).await.is_ok() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), done_rx).await;
        }
    }

    /// Start the single background writer task draining `rx` until `shutdown`.
    pub fn start_worker(self: &Arc<Self>, shutdown: CancellationToken) -> Option<JoinHandle<()>> {
        let mut rx = self.rx.lock().take()?;
        self.worker_started.store(true, Ordering::SeqCst);
        let recorder = self.clone();
        Some(tokio::spawn(async move {
            let mut batch: Vec<FlightRecord> = Vec::new();
            let mut ticker = tokio::time::interval(std::time::Duration::from_millis(250));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => {
                        while let Ok(msg) = rx.try_recv() {
                            match msg {
                                RecorderMsg::Record(rec) => batch.push(*rec),
                                RecorderMsg::Flush(tx) => {
                                    recorder.write_batch(&mut batch);
                                    let _ = tx.send(());
                                }
                            }
                        }
                        recorder.write_batch(&mut batch);
                        break;
                    }
                    _ = ticker.tick() => {
                        recorder.write_batch(&mut batch);
                    }
                    item = rx.recv() => {
                        match item {
                            Some(RecorderMsg::Record(rec)) => {
                                batch.push(*rec);
                                if batch.len() >= 16 {
                                    recorder.write_batch(&mut batch);
                                }
                            }
                            Some(RecorderMsg::Flush(tx)) => {
                                recorder.write_batch(&mut batch);
                                let _ = tx.send(());
                            }
                            None => {
                                recorder.write_batch(&mut batch);
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    fn write_batch(&self, batch: &mut Vec<FlightRecord>) {
        if batch.is_empty() {
            return;
        }
        let cfg = self.config.load_full();
        let state_dir = self.state_dir.load_full();
        if !cfg.enabled {
            batch.clear();
            return;
        }
        let path = resolve_flight_path(&cfg, state_dir.as_deref());
        let records = std::mem::take(batch);
        if let Err(error) = append_and_rotate(&path, &records, cfg.max_records.max(1)) {
            tracing::warn!(
                path = %path.display(),
                error = %redact_text(&error.to_string()),
                "failed to persist flight recorder entries"
            );
        }
    }
}

/// Resolve the JSONL flight path for `(config, defaults.state_dir)`.
pub fn resolve_flight_path(config: &FlightRecorderConfig, state_dir: Option<&Path>) -> PathBuf {
    if let Some(path) = &config.path {
        return path.clone();
    }
    match state_dir {
        Some(dir) => dir.join("flight.jsonl"),
        None => std::env::temp_dir().join("vouch-flight.jsonl"),
    }
}

/// Load recorded flight entries from disk, ignoring blank lines.
pub fn load_flight_records(path: &Path) -> Result<Vec<FlightRecord>> {
    let file = std::fs::File::open(path).map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot read flight log {}: {error}", path.display()),
        ))
    })?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("error reading {} line {}: {error}", path.display(), idx + 1),
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: FlightRecord = serde_json::from_str(trimmed).map_err(|error| {
            GatewayError::BadRequest(format!(
                "invalid flight record at {} line {}: {error}",
                path.display(),
                idx + 1
            ))
        })?;
        records.push(record);
    }
    Ok(records)
}

fn sanitize_record(record: &mut FlightRecord) {
    record.requested_model = redact_text(&record.requested_model);
    record.resolved_route = redact_text(&record.resolved_route);
    record.winning_backend = redact_text(&record.winning_backend);
    if let Some(sys) = &mut record.system {
        *sys = redact_text(sys);
    }
    for msg in &mut record.normalized_messages {
        sanitize_message(msg);
    }
}

fn sanitize_message(msg: &mut ChatMessage) {
    match &mut msg.content {
        MessageContent::Text(text) => {
            *text = redact_text(text);
        }
        MessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        *text = redact_text(text);
                    }
                    ContentPart::ImageUrl { url } => {
                        if url.starts_with("data:") {
                            *url = "[redacted-data-uri]".into();
                        } else {
                            *url = redact_text(url);
                        }
                    }
                    ContentPart::ImageBase64 { data, media_type } => {
                        *data = format!("[redacted-image:{media_type}]");
                    }
                }
            }
        }
    }
    for call in &mut msg.tool_calls {
        call.arguments = redact_text(&call.arguments);
    }
}

fn append_and_rotate(path: &Path, new_records: &[FlightRecord], max_records: usize) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut existing_lines: Vec<String> = match std::fs::File::open(path) {
        Ok(file) => BufReader::new(file)
            .lines()
            .map_while(std::result::Result::ok)
            .filter(|l| !l.trim().is_empty())
            .collect(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(err) => return Err(err.into()),
    };

    let mut serialized = Vec::with_capacity(new_records.len());
    for rec in new_records {
        let json = serde_json::to_string(rec)
            .map_err(|error| GatewayError::Internal(format!("serialize flight record: {error}")))?;
        serialized.push(json);
    }

    if existing_lines.len() + serialized.len() <= max_records {
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(path)?;
        for line in &serialized {
            writeln!(file, "{line}")?;
        }
        file.flush()?;
        return Ok(());
    }

    existing_lines.extend(serialized);
    let drop_count = existing_lines.len().saturating_sub(max_records);
    let kept = &existing_lines[drop_count..];

    let tmp_name = format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("flight.jsonl"),
        std::process::id()
    );
    let tmp_path = path.with_file_name(tmp_name);
    {
        let mut opts = OpenOptions::new();
        opts.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp_path)?;
        for line in kept {
            writeln!(file, "{line}")?;
        }
        file.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn redacts_secrets_and_rotates_jsonl_at_max_records() {
        let dir = tempfile::tempdir().expect("temp");
        let log_path = dir.path().join("flight.jsonl");

        let recorder = FlightRecorder::new(
            FlightRecorderConfig {
                enabled: true,
                max_records: 3,
                path: Some(log_path.clone()),
            },
            None,
        );
        let shutdown = CancellationToken::new();
        let worker = recorder.start_worker(shutdown.clone()).expect("worker");

        for idx in 0..5 {
            recorder.record(FlightRecord {
                timestamp: "2026-11-28T00:00:00Z".into(),
                protocol: "/v1/chat/completions".into(),
                requested_model: "coding/default".into(),
                resolved_route: "coding/default".into(),
                winning_backend: "mock".into(),
                profile_id: None,
                fallback_used: false,
                cache_outcome: "miss".into(),
                ttft_ms: 10 + idx,
                total_ms: 20 + idx,
                status: 200,
                prompt_tokens: 5,
                completion_tokens: 5,
                system: None,
                normalized_messages: vec![ChatMessage::text(
                    "user",
                    format!("msg {idx} token=sk-testsecret9876543210"),
                )],
                tools: Vec::new(),
                response_format: None,
            });
        }
        recorder.flush().await;
        shutdown.cancel();
        let _ = worker.await;

        let loaded = load_flight_records(&log_path).expect("load");
        assert_eq!(loaded.len(), 3);
        let text = loaded[2].normalized_messages[0].content.as_plain_text();
        assert!(text.contains("msg 4"));
        assert!(!text.contains("sk-testsecret9876543210"));
    }
}
