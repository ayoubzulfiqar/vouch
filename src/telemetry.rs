//! Structured logging with automatic secret redaction.
//!
//! Tracing events are rendered as JSON and passed through a writer that
//! removes bearer tokens, OpenAI-style `sk-` keys, and every configured
//! secret before the bytes reach stdout. The redactor is also available as a
//! plain function for error messages that are returned to clients.

use std::{
    io::{self, Write},
    sync::{Arc, LazyLock, OnceLock},
};

use arc_swap::ArcSwap;
use regex::Regex;
use tracing_subscriber::{EnvFilter, fmt::MakeWriter};

use crate::error::{GatewayError, Result};

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
        output
    }
}

static BEARER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{6,}").expect("static bearer regex")
});
static SK_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk-[A-Za-z0-9_-]{6,}").expect("static sk regex"));
static JSON_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)("(?:api[_-]?key|token|authorization|secret|password|credential)"\s*:\s*")[^"]*(")"#,
    )
    .expect("static json secret regex")
});
static HEADER_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(x-api-key\s*[:=]\s*)[^\s,;"']+"#).expect("static header regex")
});

/// Patterns applied after literal secrets.
///
/// Each pattern must expose the matched prefix in group 1 and an optional
/// suffix in group 2 so that a single replacement template works.
static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        BEARER.clone(),
        SK_KEY.clone(),
        JSON_SECRET.clone(),
        HEADER_SECRET.clone(),
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
}
