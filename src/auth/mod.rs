//! Ingress authentication and local environment discovery.
//!
//! [`tokens`] implements the optional virtual-token layer. [`discovery`]
//! contains the read-only probes used by `vouch doctor`.

pub mod discovery;
pub mod tokens;

pub use tokens::{Principal, TokenPolicy, TokenRegistry};
