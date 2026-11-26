## Summary

<!-- What changed and why. Keep it short; put details in the linked issue. -->

## Related issue

<!-- Link the issue this PR addresses, e.g. "Closes #123". -->

Closes: #

## Checklist

- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --all-targets -- -D warnings` passes.
- [ ] `cargo test` passes (unit + integration tests).
- [ ] `cargo build` passes.
- [ ] New behavior has tests, including failure/failover paths where relevant.
- [ ] Documentation updated (`README.md`) if behavior or CLI changed.
- [ ] `vouch.example.toml` updated if a config field or default changed, and the
      example still parses with `GatewayConfig::from_toml_str`.
- [ ] Changelog/release note added or requested for a user-visible change.
- [ ] Security review done: no tokens, keys, prompts, or auth-file contents in
      code, tests, docs, logs, or this PR; no new unauthenticated exposure of
      sensitive data.
- [ ] Invariants respected where applicable: no `unwrap`/`expect` in runtime
      paths, channels/buffers bounded, subprocesses PGID-supervised with
      `ProcessGroupGuard`, and `reqwest` still on `rustls` (not `rustls-tls`).
- [ ] Linked issue above, or an explanation why no issue exists.
- [ ] Commits are DCO signed off (`git commit -s`), with a `Signed-off-by:` line.

## Testing notes

<!-- Commands run, manual checks, and any environment details. -->

## Security considerations

<!-- If this PR touches auth, secrets, channel bounds, PGID supervision,
     reqwest features, or the unauthenticated endpoints, explain the impact.
     Otherwise write "None". -->
