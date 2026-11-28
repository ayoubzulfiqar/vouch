//! Structured logging with automatic secret redaction.
//!
//! Tracing events are rendered as JSON and passed through a writer that
//! removes bearer tokens, OpenAI-style `sk-` keys, and every configured
//! secret before the bytes reach stdout. The redactor is also available as a
//! plain function for error messages that are returned to clients.

use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::{Arc, LazyLock, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwap;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use parking_lot::Mutex as ParkingMutex;
use regex::Regex;
use serde_json::{Value, json};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{EnvFilter, fmt::MakeWriter};
use uuid::Uuid;

use crate::{
    config::TelemetryConfig,
    error::{GatewayError, Result},
};

/// Holds the current set of literal secrets and the structural patterns that
/// identify credentials.
#[derive(Debug)]
pub struct RedactionRegistry {
    secrets: ArcSwap<Vec<String>>,
}

impl RedactionRegistry {
    /// Create a registry with an initial secret list.
    pub fn new<I, S>(secrets: I) -> Arc<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let registry = Self {
            secrets: ArcSwap::from_pointee(Vec::new()),
        };
        registry.set_secrets(secrets);
        Arc::new(registry)
    }

    /// Replace the literal secret list. Values shorter than eight bytes are
    /// ignored so that common environment values do not mangle log lines.
    pub fn set_secrets<I, S>(&self, secrets: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut values: Vec<String> = secrets
            .into_iter()
            .map(Into::into)
            .filter(|value| value.len() >= 8)
            .collect();
        values.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        values.dedup();
        self.secrets.store(Arc::new(values));
    }

    /// Redact credentials from arbitrary text.
    pub fn redact(&self, input: &str) -> String {
        let mut output = input.to_string();
        for secret in self.secrets.load().iter() {
            if output.contains(secret.as_str()) {
                output = output.replace(secret.as_str(), "[REDACTED]");
            }
        }
        for regex in PATTERNS.iter() {
            output = regex.replace_all(&output, "$1[REDACTED]$2").into_owned();
        }
        BASE64_IMAGE
            .replace_all(&output, "$1[REDACTED_BYTES]")
            .into_owned()
    }
}

static BEARER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{6,}").expect("static bearer regex")
});
static SK_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk-[A-Za-z0-9_-]{6,}").expect("static sk regex"));
static VOUCH_LIVE_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"vouch_live_[A-Za-z0-9_-]{6,}").expect("static vouch key regex"));
static JSON_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)("(?:api[_-]?key|token|access[_-]?token|refresh[_-]?token|authorization|secret|password|credential)"\s*:\s*")[^"]*(")"#,
    )
    .expect("static json secret regex")
});
static ESCAPED_JSON_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(\\"(?:api[_-]?key|token|access[_-]?token|refresh[_-]?token|authorization|secret|password|credential)\\"\s*:\s*\\")(?:[^"\\]|\\[^"])*(\\")"#,
    )
    .expect("static escaped json secret regex")
});
static HEADER_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(x-api-key\s*[:=]\s*)[^\s,;"'\\]+"#).expect("static header regex")
});
static URL_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)([?&](?:key|api[_-]?key|token|access[_-]?token|refresh[_-]?token|secret)=)[^&\s"'\\]*"#,
    )
    .expect("static url query secret regex")
});
static BASE64_IMAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(data:image/[A-Za-z0-9.+-]+;base64,)[A-Za-z0-9+/=_-]+")
        .expect("static base64 image regex")
});

/// Patterns applied after literal secrets.
///
/// Each pattern must expose the matched prefix in group 1 and an optional
/// suffix in group 2 so that a single replacement template works.
static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        BEARER.clone(),
        SK_KEY.clone(),
        VOUCH_LIVE_KEY.clone(),
        ESCAPED_JSON_SECRET.clone(),
        JSON_SECRET.clone(),
        HEADER_SECRET.clone(),
        URL_SECRET.clone(),
    ]
});

static GLOBAL: OnceLock<Arc<RedactionRegistry>> = OnceLock::new();

/// Process-wide redaction registry used by logs and client-visible errors.
pub fn global_registry() -> &'static Arc<RedactionRegistry> {
    GLOBAL.get_or_init(|| RedactionRegistry::new(Vec::<String>::new()))
}

/// Redact credentials from text using the process-wide registry.
pub fn redact_text(input: &str) -> String {
    global_registry().redact(input)
}

/// Update the process-wide literal secret list.
pub fn set_global_secrets<I, S>(secrets: I)
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    global_registry().set_secrets(secrets);
}

/// Output stream used by [`RedactingMakeWriter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogTarget {
    /// Standard output (default for `vouch serve`).
    Stdout,
    /// Standard error (required for `vouch mcp --transport stdio`).
    Stderr,
}

/// Install the global JSON tracing subscriber writing to stdout.
///
/// Returns an error when a subscriber has already been installed, which makes
/// double-initialization in tests explicit rather than silently ignored.
pub fn init(log_level: &str, secrets: impl IntoIterator<Item = String>) -> Result<()> {
    init_with_target(log_level, secrets, LogTarget::Stdout)
}

/// Install the global JSON tracing subscriber writing to stderr.
pub fn init_stderr(log_level: &str, secrets: impl IntoIterator<Item = String>) -> Result<()> {
    init_with_target(log_level, secrets, LogTarget::Stderr)
}

/// Install the global JSON tracing subscriber with an explicit output target.
pub fn init_with_target(
    log_level: &str,
    secrets: impl IntoIterator<Item = String>,
    target: LogTarget,
) -> Result<()> {
    set_global_secrets(secrets);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_ansi(false)
        .with_current_span(false)
        .with_span_list(false)
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_writer(RedactingMakeWriter::new(global_registry().clone(), target))
        .with_env_filter(filter)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|error| GatewayError::Internal(format!("cannot install tracing: {error}")))
}

/// `MakeWriter` that routes every formatted event through the redactor.
#[derive(Clone, Debug)]
pub struct RedactingMakeWriter {
    registry: Arc<RedactionRegistry>,
    target: LogTarget,
}

impl RedactingMakeWriter {
    /// Build a writer for the given registry and output target.
    pub fn new(registry: Arc<RedactionRegistry>, target: LogTarget) -> Self {
        Self { registry, target }
    }
}

impl<'a> MakeWriter<'a> for RedactingMakeWriter {
    type Writer = RedactingWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter {
            registry: &self.registry,
            target: self.target,
            buffer: Vec::with_capacity(512),
        }
    }
}

/// Buffers a complete log event, redacts it, and writes it to the target stream.
pub struct RedactingWriter<'a> {
    registry: &'a RedactionRegistry,
    target: LogTarget,
    buffer: Vec<u8>,
}

impl RedactingWriter<'_> {
    fn flush_buffer(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let raw = String::from_utf8_lossy(&self.buffer);
        let redacted = self.registry.redact(&raw);
        self.buffer.clear();
        match self.target {
            LogTarget::Stdout => {
                let stdout = io::stdout();
                let mut lock = stdout.lock();
                lock.write_all(redacted.as_bytes())?;
                lock.flush()
            }
            LogTarget::Stderr => {
                let stderr = io::stderr();
                let mut lock = stderr.lock();
                lock.write_all(redacted.as_bytes())?;
                lock.flush()
            }
        }
    }
}

impl Write for RedactingWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()
    }
}

impl Drop for RedactingWriter<'_> {
    fn drop(&mut self) {
        let _ = self.flush_buffer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_bearer_and_sk_keys() {
        let registry = RedactionRegistry::new(Vec::<String>::new());
        let text = "Authorization: Bearer abcdefghijklmnop and key sk-1234567890abcdef";
        let redacted = registry.redact(text);
        assert!(!redacted.contains("abcdefghijklmnop"));
        assert!(!redacted.contains("1234567890abcdef"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn redacts_configured_literals() {
        let registry = RedactionRegistry::new(["vouch-super-secret-value"]);
        let redacted = registry.redact("token=vouch-super-secret-value done");
        assert_eq!(redacted, "token=[REDACTED] done");
    }

    #[test]
    fn redacts_json_fields() {
        let registry = RedactionRegistry::new(Vec::<String>::new());
        let redacted = registry.redact(r#"{"api_key":"abcdefgh","token":"xyz12345"}"#);
        assert!(!redacted.contains("abcdefgh"));
        assert!(!redacted.contains("xyz12345"));
    }

    #[test]
    fn ignores_short_secrets() {
        let registry = RedactionRegistry::new(["abc"]);
        assert_eq!(registry.redact("abc stays"), "abc stays");
    }

    #[test]
    fn redacts_url_query_params_and_oauth_tokens() {
        let registry = RedactionRegistry::new(Vec::<String>::new());
        let url = "GET https://api.example.com/v1?key=secret_url_key_123&api_key=ak_999&token=tok_abc&foo=bar";
        let redacted_url = registry.redact(url);
        assert!(!redacted_url.contains("secret_url_key_123"));
        assert!(!redacted_url.contains("ak_999"));
        assert!(!redacted_url.contains("tok_abc"));
        assert!(redacted_url.contains("&foo=bar"));

        let json = r#"{"access_token":"acc_secret_val","refresh_token":"ref_secret_val","ok":1}"#;
        let redacted_json = registry.redact(json);
        assert!(!redacted_json.contains("acc_secret_val"));
        assert!(!redacted_json.contains("ref_secret_val"));
        assert!(redacted_json.contains("\"ok\":1"));
    }

    #[test]
    fn redacts_escaped_quotes_vouch_keys_and_base64_images() {
        let registry = RedactionRegistry::new(Vec::<String>::new());
        let escaped = r#"upstream payload: \"Bearer tok_123456\" and \"sk-abcdef123456\" and \"vouch_live_supersecret99\" and \"api_key\":\"hidden_nested_key\" with data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="tail""#;
        let redacted = registry.redact(escaped);
        assert!(!redacted.contains("tok_123456"));
        assert!(!redacted.contains("sk-abcdef123456"));
        assert!(!redacted.contains("vouch_live_supersecret99"));
        assert!(!redacted.contains("hidden_nested_key"));
        assert!(!redacted.contains("iVBORw0KGgoAAAANSUhEUg=="));
        assert!(redacted.contains("data:image/png;base64,[REDACTED_BYTES]"));
    }

    #[test]
    fn parses_w3c_traceparent_and_creates_child_spans() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        headers.insert("tracestate", HeaderValue::from_static("vendor=foo"));

        let ctx = TraceContext::from_headers(&headers);
        assert_eq!(ctx.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(ctx.parent_span_id.as_deref(), Some("00f067aa0ba902b7"));
        assert_eq!(ctx.tracestate.as_deref(), Some("vendor=foo"));

        let child = ctx.child();
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_eq!(child.parent_span_id.as_deref(), Some(ctx.span_id.as_str()));
        assert_ne!(child.span_id, ctx.span_id);
        assert!(
            child
                .traceparent()
                .starts_with("00-4bf92f3577b34da6a3ce929d0e0e4736-")
        );
    }

    #[test]
    fn otlp_payload_redacts_secrets_from_attributes() {
        set_global_secrets(["super-secret-otel-token"]);
        let span = OtlpSpanRecord {
            trace_id: "4bf92f3577b34da6a3ce929d0e0e4736".into(),
            span_id: "00f067aa0ba902b7".into(),
            parent_span_id: None,
            name: "ingress",
            kind: SpanKind::Server,
            start_nanos: 1_000,
            end_nanos: 2_000,
            attributes: vec![(
                "vouch.detail",
                SpanAttrValue::String("token=super-secret-otel-token Bearer secret9999".into()),
            )],
            error_message: None,
        };
        let payload = build_otlp_payload("vouch-test", &[span]);
        let serialized = payload.to_string();
        assert!(!serialized.contains("super-secret-otel-token"));
        assert!(!serialized.contains("secret9999"));
        assert!(serialized.contains("[REDACTED]"));
    }
}

// ---------------------------------------------------------------------------
// W3C TraceContext & OpenTelemetry OTLP/HTTP JSON span exporter
// ---------------------------------------------------------------------------

const OTLP_QUEUE_CAPACITY: usize = 1024;
const OTLP_BATCH_SIZE: usize = 32;

/// OpenTelemetry span kind constants (`Span.SpanKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// Internal operation (`router.dispatch`, `workspace.worktree`).
    Internal = 1,
    /// Server ingress (`ingress`).
    Server = 2,
    /// Outbound client call (`driver.exec`).
    Client = 3,
}

/// Propagated W3C trace context (`traceparent` + optional `tracestate`).
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// 32-char lowercase hex trace ID.
    pub trace_id: String,
    /// 16-char lowercase hex span ID for the current span.
    pub span_id: String,
    /// Optional parent 16-char hex span ID from caller or parent span.
    pub parent_span_id: Option<String>,
    /// 2-char hex trace flags (`01` sampled).
    pub trace_flags: String,
    /// Optional W3C `tracestate` header value.
    pub tracestate: Option<String>,
}

impl TraceContext {
    /// Extract W3C `traceparent`/`tracestate` from HTTP headers, generating a
    /// fresh non-zero trace/span ID pair when missing or invalid.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let traceparent = headers
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_w3c_traceparent);
        let tracestate = headers
            .get("tracestate")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|v| !v.is_empty() && v.len() <= 512)
            .map(ToString::to_string);

        match traceparent {
            Some((trace_id, remote_span, trace_flags)) => Self {
                trace_id,
                span_id: random_span_id(),
                parent_span_id: Some(remote_span),
                trace_flags,
                tracestate,
            },
            None => Self {
                trace_id: random_trace_id(),
                span_id: random_span_id(),
                parent_span_id: None,
                trace_flags: "01".into(),
                tracestate,
            },
        }
    }

    /// Extract trace context from optional MCP JSON-RPC `params._meta` first,
    /// falling back to HTTP headers or generating a new root context.
    pub fn from_headers_and_meta(headers: Option<&HeaderMap>, meta: Option<&Value>) -> Self {
        if let Some(meta) = meta
            && let Some(tp) = meta.get("traceparent").and_then(Value::as_str)
            && let Some((trace_id, remote_span, trace_flags)) = parse_w3c_traceparent(tp)
        {
            let tracestate = meta
                .get("tracestate")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty() && v.len() <= 512)
                .map(ToString::to_string)
                .or_else(|| {
                    headers
                        .and_then(|h| h.get("tracestate"))
                        .and_then(|v| v.to_str().ok())
                        .map(str::trim)
                        .filter(|v| !v.is_empty())
                        .map(ToString::to_string)
                });
            return Self {
                trace_id,
                span_id: random_span_id(),
                parent_span_id: Some(remote_span),
                trace_flags,
                tracestate,
            };
        }
        match headers {
            Some(headers) => Self::from_headers(headers),
            None => Self {
                trace_id: random_trace_id(),
                span_id: random_span_id(),
                parent_span_id: None,
                trace_flags: "01".into(),
                tracestate: None,
            },
        }
    }

    /// Create a child context whose `parent_span_id` is this context's `span_id`.
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: random_span_id(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: self.trace_flags.clone(),
            tracestate: self.tracestate.clone(),
        }
    }

    /// Render W3C `traceparent` header string.
    pub fn traceparent(&self) -> String {
        format!("00-{}-{}-{}", self.trace_id, self.span_id, self.trace_flags)
    }
}

/// Attribute value carried on an OTLP span record.
#[derive(Debug, Clone)]
pub enum SpanAttrValue {
    /// UTF-8 string value (redacted before export).
    String(String),
    /// Signed integer metric/count value.
    Int(i64),
    /// Boolean flag value.
    Bool(bool),
}

/// Completed trace span queued for background OTLP/HTTP export.
#[derive(Debug, Clone)]
pub struct OtlpSpanRecord {
    /// 32-char hex trace ID.
    pub trace_id: String,
    /// 16-char hex span ID.
    pub span_id: String,
    /// Optional 16-char hex parent span ID.
    pub parent_span_id: Option<String>,
    /// Span operation name (`ingress`, `router.dispatch`, `driver.exec`, `workspace.worktree`).
    pub name: &'static str,
    /// Span kind.
    pub kind: SpanKind,
    /// Start timestamp in UNIX epoch nanoseconds.
    pub start_nanos: u128,
    /// End timestamp in UNIX epoch nanoseconds.
    pub end_nanos: u128,
    /// Redacted structured attributes.
    pub attributes: Vec<(&'static str, SpanAttrValue)>,
    /// Optional redacted error status message.
    pub error_message: Option<String>,
}

/// In-flight span builder created via [`OtelExporter::start_span`].
#[derive(Debug)]
pub struct ActiveSpan {
    exporter: Option<Arc<OtelExporter>>,
    context: TraceContext,
    name: &'static str,
    kind: SpanKind,
    start_nanos: u128,
    attributes: Vec<(&'static str, SpanAttrValue)>,
}

impl ActiveSpan {
    /// Span's W3C context (pass `.context()` to children).
    pub fn context(&self) -> &TraceContext {
        &self.context
    }

    /// Attach a string attribute (automatically redacted before export).
    pub fn set_str(&mut self, key: &'static str, value: impl AsRef<str>) {
        if self.exporter.is_some() {
            self.attributes
                .push((key, SpanAttrValue::String(redact_text(value.as_ref()))));
        }
    }

    /// Attach a signed integer attribute.
    pub fn set_int(&mut self, key: &'static str, value: i64) {
        if self.exporter.is_some() {
            self.attributes.push((key, SpanAttrValue::Int(value)));
        }
    }

    /// Attach a boolean attribute.
    pub fn set_bool(&mut self, key: &'static str, value: bool) {
        if self.exporter.is_some() {
            self.attributes.push((key, SpanAttrValue::Bool(value)));
        }
    }

    /// Complete the span successfully and enqueue for async OTLP export.
    pub fn finish_ok(self) {
        self.finish(None);
    }

    /// Complete the span with a redacted error description and enqueue for export.
    pub fn finish_err(self, message: &str) {
        self.finish(Some(redact_text(message)));
    }

    fn finish(self, error_message: Option<String>) {
        let Some(exporter) = self.exporter else {
            return;
        };
        let end_nanos = now_unix_nanos().max(self.start_nanos);
        exporter.enqueue(OtlpSpanRecord {
            trace_id: self.context.trace_id,
            span_id: self.context.span_id,
            parent_span_id: self.context.parent_span_id,
            name: self.name,
            kind: self.kind,
            start_nanos: self.start_nanos,
            end_nanos,
            attributes: self.attributes,
            error_message,
        });
    }
}

#[derive(Debug, Clone)]
struct OtlpEndpointConfig {
    endpoint: String,
    service_name: String,
    headers: BTreeMap<String, String>,
}

/// Non-blocking, bounded OpenTelemetry OTLP/HTTP JSON trace exporter.
#[derive(Debug)]
pub struct OtelExporter {
    config: ArcSwap<Option<OtlpEndpointConfig>>,
    tx: mpsc::Sender<OtlpSpanRecord>,
    rx: ParkingMutex<Option<mpsc::Receiver<OtlpSpanRecord>>>,
    client: reqwest::Client,
}

impl OtelExporter {
    /// Build a trace exporter configured from `[telemetry]`.
    pub fn new(config: &TelemetryConfig) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(OTLP_QUEUE_CAPACITY);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        let exporter = Arc::new(Self {
            config: ArcSwap::from_pointee(None),
            tx,
            rx: ParkingMutex::new(Some(rx)),
            client,
        });
        exporter.reconfigure(config);
        exporter
    }

    /// Atomically update `[telemetry]` settings on configuration reload.
    pub fn reconfigure(&self, config: &TelemetryConfig) {
        let next = config.otlp_endpoint.as_ref().map(|ep| OtlpEndpointConfig {
            endpoint: ep.clone(),
            service_name: config.service_name.clone(),
            headers: config.otlp_headers.clone(),
        });
        self.config.store(Arc::new(next));
    }

    /// Whether an OTLP endpoint is currently configured.
    pub fn is_enabled(&self) -> bool {
        self.config.load().is_some()
    }

    /// Start a recorded span under `context`. When `otlp_endpoint` is unset this
    /// stays inert without allocating attributes or queuing spans.
    pub fn start_span(
        self: &Arc<Self>,
        context: TraceContext,
        name: &'static str,
        kind: SpanKind,
    ) -> ActiveSpan {
        let exporter = self.is_enabled().then(|| self.clone());
        ActiveSpan {
            exporter,
            context,
            name,
            kind,
            start_nanos: now_unix_nanos(),
            attributes: Vec::new(),
        }
    }

    fn enqueue(&self, span: OtlpSpanRecord) {
        if self.tx.try_send(span).is_err() {
            tracing::debug!("OTLP span queue full; dropping span to keep bounded memory");
        }
    }

    /// Spawn the background batch exporter task. Safe to call once; flushes
    /// remaining spans on `shutdown`.
    pub fn start_worker(self: &Arc<Self>, shutdown: CancellationToken) -> Option<JoinHandle<()>> {
        let mut rx = self.rx.lock().take()?;
        let exporter = self.clone();
        Some(tokio::spawn(async move {
            let mut batch = Vec::with_capacity(OTLP_BATCH_SIZE);
            let mut ticker = tokio::time::interval(Duration::from_secs(1));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => {
                        while let Ok(span) = rx.try_recv() {
                            batch.push(span);
                            if batch.len() >= OTLP_BATCH_SIZE {
                                exporter.flush_batch(&mut batch).await;
                            }
                        }
                        if !batch.is_empty() {
                            exporter.flush_batch(&mut batch).await;
                        }
                        break;
                    }
                    _ = ticker.tick() => {
                        if !batch.is_empty() {
                            exporter.flush_batch(&mut batch).await;
                        }
                    }
                    item = rx.recv() => {
                        match item {
                            Some(span) => {
                                batch.push(span);
                                if batch.len() >= OTLP_BATCH_SIZE {
                                    exporter.flush_batch(&mut batch).await;
                                }
                            }
                            None => {
                                if !batch.is_empty() {
                                    exporter.flush_batch(&mut batch).await;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn flush_batch(&self, batch: &mut Vec<OtlpSpanRecord>) {
        if batch.is_empty() {
            return;
        }
        let cfg = self.config.load_full();
        let Some(cfg) = cfg.as_ref() else {
            batch.clear();
            return;
        };
        let payload = build_otlp_payload(&cfg.service_name, batch);
        batch.clear();

        let mut req = self.client.post(&cfg.endpoint).json(&payload);
        for (name, val) in &cfg.headers {
            if let Ok(hn) = HeaderName::from_bytes(name.as_bytes())
                && let Ok(hv) = HeaderValue::from_str(val)
            {
                req = req.header(hn, hv);
            }
        }
        if let Err(error) = req.send().await {
            tracing::warn!(error = %redact_text(&error.to_string()), "failed to export OTLP traces");
        }
    }
}

fn build_otlp_payload(service_name: &str, spans: &[OtlpSpanRecord]) -> Value {
    let encoded_spans: Vec<Value> = spans
        .iter()
        .map(|span| {
            let attrs: Vec<Value> = span
                .attributes
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        SpanAttrValue::String(s) => json!({ "stringValue": redact_text(s) }),
                        SpanAttrValue::Int(i) => json!({ "intValue": i.to_string() }),
                        SpanAttrValue::Bool(b) => json!({ "boolValue": b }),
                    };
                    json!({ "key": k, "value": val })
                })
                .collect();

            let mut obj = serde_json::Map::new();
            obj.insert("traceId".into(), json!(span.trace_id));
            obj.insert("spanId".into(), json!(span.span_id));
            if let Some(parent) = &span.parent_span_id {
                obj.insert("parentSpanId".into(), json!(parent));
            }
            obj.insert("name".into(), json!(span.name));
            obj.insert("kind".into(), json!(span.kind as u8));
            obj.insert(
                "startTimeUnixNano".into(),
                json!(span.start_nanos.to_string()),
            );
            obj.insert("endTimeUnixNano".into(), json!(span.end_nanos.to_string()));
            if !attrs.is_empty() {
                obj.insert("attributes".into(), Value::Array(attrs));
            }
            if let Some(err) = &span.error_message {
                obj.insert(
                    "status".into(),
                    json!({ "code": 2, "message": redact_text(err) }),
                );
            } else {
                obj.insert("status".into(), json!({ "code": 1 }));
            }
            Value::Object(obj)
        })
        .collect();

    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    { "key": "service.name", "value": { "stringValue": redact_text(service_name) } },
                    { "key": "service.version", "value": { "stringValue": crate::VERSION } }
                ]
            },
            "scopeSpans": [{
                "scope": { "name": "vouch", "version": crate::VERSION },
                "spans": encoded_spans
            }]
        }]
    })
}

fn parse_w3c_traceparent(raw: &str) -> Option<(String, String, String)> {
    let trimmed = raw.trim().to_ascii_lowercase();
    let mut parts = trimmed.split('-');
    let version = parts.next()?;
    let trace_id = parts.next()?;
    let span_id = parts.next()?;
    let flags = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if version.len() != 2 || version == "ff" || !version.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if trace_id.len() != 32
        || !trace_id.bytes().all(|b| b.is_ascii_hexdigit())
        || trace_id.bytes().all(|b| b == b'0')
    {
        return None;
    }
    if span_id.len() != 16
        || !span_id.bytes().all(|b| b.is_ascii_hexdigit())
        || span_id.bytes().all(|b| b == b'0')
    {
        return None;
    }
    if flags.len() != 2 || !flags.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some((trace_id.to_string(), span_id.to_string(), flags.to_string()))
}

fn random_trace_id() -> String {
    let id = Uuid::new_v4();
    hex::encode(id.as_bytes())
}

fn random_span_id() -> String {
    let id = Uuid::new_v4();
    hex::encode(&id.as_bytes()[..8])
}

fn now_unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

/// Redacted summary of one completed gateway request stored in memory for
/// `GET /v1/traces` and the interactive Trace Waterfall UI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecentTraceEntry {
    /// Request correlation identifier (`x-vouch-request-id`).
    pub request_id: String,
    /// RFC-3339 UTC timestamp when completed.
    pub timestamp: String,
    /// Wire protocol (`/v1/chat/completions`, `/v1/messages`, ...).
    pub protocol: String,
    /// Resolved route key.
    pub route: String,
    /// Original model/alias requested by client.
    pub requested_model: String,
    /// Backend that handled the execution (`"none"` when rejected pre-dispatch).
    pub backend: String,
    /// Active credential profile id when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    /// True when failover or hedged backup succeeded.
    pub fallback: bool,
    /// Deterministic cache outcome (`hit`, `miss`, `bypass`, `disabled`, `coalesced`).
    pub cache: String,
    /// True when satisfied by singleflight coalescing.
    pub coalesced: bool,
    /// Time to first chunk or response completion in milliseconds.
    pub ttft_ms: u64,
    /// Total end-to-end request duration in milliseconds.
    pub total_ms: u64,
    /// HTTP status code.
    pub status: u16,
    /// Prompt BPE token count.
    pub prompt_tokens: u64,
    /// Completion BPE token count.
    pub completion_tokens: u64,
    /// Redacted error summary, if failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Bounded FIFO ring buffer storing the most recent `capacity` request traces.
#[derive(Debug)]
pub struct TraceRingBuffer {
    capacity: usize,
    entries: ParkingMutex<std::collections::VecDeque<RecentTraceEntry>>,
}

impl TraceRingBuffer {
    /// Create a ring buffer capped at `capacity` recent traces (default `100`).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            entries: ParkingMutex::new(std::collections::VecDeque::with_capacity(cap)),
        }
    }

    /// Push a sanitized trace entry, evicting the oldest if at capacity.
    pub fn push(&self, mut entry: RecentTraceEntry) {
        if let Some(err) = entry.error.take() {
            entry.error = Some(redact_text(&err));
        }
        entry.requested_model = redact_text(&entry.requested_model);
        entry.route = redact_text(&entry.route);
        let mut guard = self.entries.lock();
        if guard.len() >= self.capacity {
            guard.pop_front();
        }
        guard.push_back(entry);
    }

    /// Return recent traces ordered newest-first.
    pub fn recent(&self) -> Vec<RecentTraceEntry> {
        let guard = self.entries.lock();
        guard.iter().rev().cloned().collect()
    }
}
