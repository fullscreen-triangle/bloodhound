//! `tracker` — repo-federation tracker.
//!
//! Tracks a group of repos and their conserved sense/goal (the character invariant χ,
//! from the contact-graph foundation / split-attention T1), keeps a repo's several
//! origins in step with per-origin visibility (`sync`), and holds one profile of you
//! across every forge (`profile`). It composes other installable CLIs — `purpose`,
//! `git`, `curl`, network-yield — and reimplements none.
//!
//! Three ways in, one core:
//!
//! * **people and scripts:** the `tracker` CLI, with `--json` on every command;
//! * **AI agents:** `tracker mcp`, a Model Context Protocol server over stdio;
//! * **Rust:** [`api`], the same questions and actions as plain functions.
//!
//! See `thrust/docs/tracker/repo-federation-tracker-design.md`.

pub mod agent;
pub mod api;
pub mod error;
pub mod mcp;

#[doc(hidden)]
pub mod chi;
#[doc(hidden)]
pub mod cli;
#[doc(hidden)]
pub mod gitops;
#[doc(hidden)]
pub mod graph;
#[doc(hidden)]
pub mod profile;
#[doc(hidden)]
pub mod purpose;
#[doc(hidden)]
pub mod registry;
#[doc(hidden)]
pub mod serve;
#[doc(hidden)]
pub mod sync;
#[doc(hidden)]
pub mod tokens;
