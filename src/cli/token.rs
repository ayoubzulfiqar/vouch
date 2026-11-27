//! SPDX-License-Identifier: GPL-3.0-or-later
//! Token generation utility (`vouch token create`).

use std::io::Write;

use crate::{
    cli::{TokenArgs, TokenCommand},
    error::{GatewayError, Result},
};

/// Execute `vouch token ...`.
pub fn run(args: &TokenArgs) -> Result<()> {
    match &args.command {
        TokenCommand::Create(create) => {
            if create.rpm == Some(0) {
                return Err(GatewayError::Config(
                    "--rpm must be greater than zero".into(),
                ));
            }
            let snippet = generate_token_toml(create.name.as_deref(), create.rpm, &create.allow)?;
            let mut stdout = std::io::stdout().lock();
            write!(stdout, "{snippet}")?;
            Ok(())
        }
    }
}

/// Generate a cryptographically random 256-bit `vouch_live_...` token and
/// format it as a ready-to-paste `[[tokens]]` TOML section.
pub fn generate_token_toml(
    name: Option<&str>,
    rpm: Option<u32>,
    allowed_models: &[String],
) -> Result<String> {
    let token = generate_random_token()?;
    let mut out = String::new();
    out.push_str("[[tokens]]\n");
    out.push_str(&format!("token = {:?}\n", token));
    if let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) {
        out.push_str(&format!("name = {:?}\n", name));
    }
    if let Some(rpm) = rpm {
        out.push_str(&format!("rpm = {rpm}\n"));
    }
    if !allowed_models.is_empty() {
        let items: Vec<String> = allowed_models
            .iter()
            .map(|pat| format!("{pat:?}"))
            .collect();
        out.push_str(&format!("allowed_models = [{}]\n", items.join(", ")));
    }
    Ok(out)
}

/// Generate a 256-bit random token formatted as `vouch_live_<64_hex_chars>`.
pub fn generate_random_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| {
        GatewayError::Internal(format!("CSPRNG failed generating token: {error}"))
    })?;
    Ok(format!("vouch_live_{}", hex::encode(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;

    #[test]
    fn generates_valid_256_bit_token_and_toml_block() {
        let raw = generate_random_token().expect("random token");
        assert!(raw.starts_with("vouch_live_"));
        assert_eq!(raw.len(), "vouch_live_".len() + 64);

        let block = generate_token_toml(Some("orchestrator"), Some(120), &["coding/*".into()])
            .expect("render toml block");
        let doc = format!(
            r#"
[[backends]]
id = "b"
kind = "cli"
command = "/bin/echo"
stdin = true

[routes]
"coding/default" = ["b"]

{block}
"#
        );
        let cfg = GatewayConfig::from_toml_str(&doc).expect("generated block is valid TOML");
        assert_eq!(cfg.tokens.len(), 1);
        assert_eq!(cfg.tokens[0].name.as_deref(), Some("orchestrator"));
        assert_eq!(cfg.tokens[0].rpm, Some(120));
        assert!(cfg.tokens[0].token.starts_with("vouch_live_"));
        assert_eq!(cfg.tokens[0].token.len(), 75);
    }
}
