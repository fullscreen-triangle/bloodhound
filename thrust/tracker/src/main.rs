//! `tracker` — repo-federation tracker.
//!
//! Tracks a group of repos and their conserved sense/goal (the character invariant χ,
//! from the contact-graph foundation / split-attention T1). It composes two other
//! installable Rust CLIs and reimplements neither:
//!
//!   * `purpose`      — the *search organ* (search-not-fetch, invariant I3);
//!   * network-yield  — the *execution organ* (running repo code in the cloud).
//!
//! See `thrust/docs/tracker/repo-federation-tracker-design.md`.

mod chi;
mod cli;
mod error;
mod purpose;
mod registry;

use clap::Parser;

fn main() -> std::process::ExitCode {
    let cli = cli::Cli::parse();
    match cli::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tracker: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
