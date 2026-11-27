//! SPDX-License-Identifier: GPL-3.0-or-later
//! BPE token accounting via `tiktoken-rs` for CLI and PTY backends.

use std::sync::OnceLock;

use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

use crate::driver::Usage;

static CL100K: OnceLock<Option<CoreBPE>> = OnceLock::new();
static O200K: OnceLock<Option<CoreBPE>> = OnceLock::new();

fn encoder_for_model(model: &str) -> Option<&'static CoreBPE> {
    let lower = model.to_ascii_lowercase();
    let prefer_o200k = lower.contains("gpt-4o")
        || lower.contains("gpt-5")
        || lower.contains("o1")
        || lower.contains("o3")
        || lower.contains("o4");
    if prefer_o200k && let Some(bpe) = O200K.get_or_init(|| o200k_base().ok()).as_ref() {
        return Some(bpe);
    }
    CL100K
        .get_or_init(|| cl100k_base().ok())
        .as_ref()
        .or_else(|| O200K.get_or_init(|| o200k_base().ok()).as_ref())
}

/// Count BPE tokens in `text` for `model`, falling back safely if BPE init fails.
pub fn count_tokens(model: &str, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    match encoder_for_model(model) {
        Some(bpe) => bpe.encode_with_special_tokens(text).len() as u64,
        None => (text.chars().count() as u64).div_ceil(4),
    }
}

/// Estimate prompt and completion token usage for `model`.
pub fn estimate_usage(model: &str, prompt: &str, completion: &str) -> Usage {
    Usage::new(count_tokens(model, prompt), count_tokens(model, completion))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_bpe_tokens_for_cl100k_and_o200k() {
        assert_eq!(count_tokens("coding/default", ""), 0);
        let cl = count_tokens("coding/default", "hello world");
        let o200k = count_tokens("gpt-4o", "hello world");
        assert!(cl >= 2);
        assert!(o200k >= 2);
    }
}
