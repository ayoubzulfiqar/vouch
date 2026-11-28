//! SPDX-License-Identifier: GPL-3.0-or-later
//! Outbound security policies and content guardrails.

pub mod egress_guard;

pub use egress_guard::{EGRESS_REDACTED_PLACEHOLDER, EgressGuard};
