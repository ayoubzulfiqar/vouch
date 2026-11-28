//! SPDX-License-Identifier: GPL-3.0-or-later
//! Cross-agent shared memory ledger, deterministic handoff context synthesis,
//! and model-free `trace` MCP intelligence bridge (`[shared_memory]`, `[trace]`).

pub mod handoff;
pub mod ledger;
pub mod trace_bridge;

pub use handoff::{SharedMemoryEngine, truncate_to_bpe_budget};
pub use ledger::{
    CanonicalTurn, HandoffEvent, HandoffReason, RecordTurnInput, SessionLedgerEntry,
    SessionLedgerStore, SessionSummary, extract_sanitized_repo_paths, sanitize_relative_repo_path,
};
pub use trace_bridge::{TraceBridge, TraceProbeStatus, resolve_workspace_root_header};
