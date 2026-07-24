//! Command-line surface and handlers.

use crate::chi;
use crate::error::{Result, TrackerError};
use crate::purpose::{self, Index};
use crate::registry::{Federation, RepoRecord};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "tracker",
    version,
    about = "Repo-federation tracker: tracks a group of repos and their conserved sense/goal (χ).",
    long_about = "Tracks not just commits but each repo's conserved sense/goal (the character \
invariant χ). Composes the `purpose` search CLI (search-not-fetch) and the network-yield \
execution CLI (running code). It reimplements neither."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Create a federation here (writes .tracker/).
    Init,

    /// Register a repo and build its self-graph via `purpose index`.
    Add {
        /// Path to the repo root.
        path: PathBuf,
        /// Name for the repo (defaults to the directory name).
        #[arg(long)]
        name: Option<String>,
        /// Optional remote URL, for federation-level identity.
        #[arg(long)]
        remote: Option<String>,
    },

    /// List tracked repos with their current χ and committed count.
    List,

    /// Report a repo's sense/goal, freshly searched (never fetched).
    ///
    /// With no question, reports the repo's standing sense (its salient surface).
    /// With a question, runs `purpose ask` against the repo's current index.
    Sense {
        repo: String,
        question: Option<String>,
    },

    /// Ask a question across the whole federation (fans `purpose ask` over every repo).
    Ask { question: String },

    /// Recompute χ and report whether the repo's sense has moved since last time.
    Drift { repo: String },

    /// Hand a goal to the execution organ (network-yield). Degrades if not installed.
    Run { repo: String, goal: String },
}

/// Entry point from `main`.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Cmd::Init => cmd_init(),
        Cmd::Add { path, name, remote } => cmd_add(path, name, remote),
        Cmd::List => cmd_list(),
        Cmd::Sense { repo, question } => cmd_sense(repo, question),
        Cmd::Ask { question } => cmd_ask(question),
        Cmd::Drift { repo } => cmd_drift(repo),
        Cmd::Run { repo, goal } => cmd_run(repo, goal),
    }
}

fn cwd() -> Result<PathBuf> {
    Ok(std::env::current_dir()?)
}

fn cmd_init() -> Result<()> {
    let root = cwd()?;
    Federation::init(&root)?;
    println!("Initialised federation at {}", root.display());
    println!("Next: `tracker add <path-to-repo>`");
    Ok(())
}

fn cmd_add(path: PathBuf, name: Option<String>, remote: Option<String>) -> Result<()> {
    let mut fed = Federation::discover(&cwd()?)?;
    let abs = path
        .canonicalize()
        .map_err(|_| TrackerError::BadRepoPath(path.clone()))?;
    if !abs.is_dir() {
        return Err(TrackerError::BadRepoPath(abs));
    }
    let name = name.unwrap_or_else(|| {
        abs.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "repo".into())
    });

    // Build the self-graph via the search organ (construction phase, I4).
    ensure_purpose()?;
    println!("Indexing {} via `purpose index`…", abs.display());
    purpose::index(&abs)?;

    // Compute χ from the fresh index.
    let index = Index::load(&abs, &name)?;
    let character = chi::compute(&index);

    let mut record = RepoRecord {
        name: name.clone(),
        path: abs,
        remote,
        chi: Some(character.chi),
        committed: 0,
    };
    record.record_act(); // the index+χ is a committed tracked act (I2)

    fed.add(record)?;
    fed.save()?;

    println!(
        "Added {name}: χ = {:.3} over {} blocks (m = 1).",
        character.chi, character.blocks
    );
    print_salient(&character);
    Ok(())
}

fn cmd_list() -> Result<()> {
    let fed = Federation::discover(&cwd()?)?;
    if fed.repos.is_empty() {
        println!("No repos tracked yet. Add one with `tracker add <path>`.");
        return Ok(());
    }
    println!("{:<24} {:>10} {:>6}  {}", "REPO", "χ", "m", "PATH");
    for r in &fed.repos {
        let chi = r.chi.map(|c| format!("{c:.3}")).unwrap_or_else(|| "—".into());
        println!(
            "{:<24} {:>10} {:>6}  {}",
            r.name,
            chi,
            r.committed,
            r.path.display()
        );
    }
    Ok(())
}

fn cmd_sense(repo: String, question: Option<String>) -> Result<()> {
    let fed = Federation::discover(&cwd()?)?;
    let record = fed.get(&repo)?;
    ensure_purpose()?;
    // I3: freshness — re-index so the search reflects the repo as it is now.
    purpose::index(&record.path)?;

    match question {
        Some(q) => {
            // Search-not-fetch: the answer is a fresh slice, never stored (I3).
            let hits = purpose::ask(&record.path, &q)?;
            report_hits(&repo, &q, &hits);
        }
        None => {
            // Standing sense: the salient surface of χ, narrated by searching for it.
            let index = Index::load(&record.path, &repo)?;
            let character = chi::compute(&index);
            println!(
                "{repo}: sense/goal — χ = {:.3} over {} blocks.",
                character.chi, character.blocks
            );
            print_salient(&character);
            println!(
                "\nCheapest conceptual split (χ severs this region of {} block(s)):",
                character.cut_side.len()
            );
            for b in character.cut_side.iter().take(12) {
                println!("  · {b}");
            }
        }
    }
    Ok(())
}

fn cmd_ask(question: String) -> Result<()> {
    let fed = Federation::discover(&cwd()?)?;
    ensure_purpose()?;
    if fed.repos.is_empty() {
        println!("No repos tracked yet.");
        return Ok(());
    }
    // Fan the search across the federation; report per-repo (search-not-fetch, I3).
    let mut any = false;
    for r in &fed.repos {
        purpose::index(&r.path)?; // freshness
        let hits = purpose::ask(&r.path, &question)?;
        if !hits.is_empty() {
            any = true;
            println!("── {} ──", r.name);
            for h in hits.iter().take(5) {
                println!("  {}:{}  [{}] {}", h.file, h.line, h.kind, h.name);
            }
        }
    }
    if !any {
        println!("No matches across the federation for {question:?}.");
    }
    Ok(())
}

fn cmd_drift(repo: String) -> Result<()> {
    let mut fed = Federation::discover(&cwd()?)?;
    ensure_purpose()?;
    let path = fed.get(&repo)?.path.clone();
    let old = fed.get(&repo)?.chi;

    // Reconstruct the current sense (construction phase, I4).
    purpose::index(&path)?;
    let index = Index::load(&path, &repo)?;
    let character = chi::compute(&index);
    let new = character.chi;

    // Commit the recompute (I2) and persist the new χ.
    {
        let record = fed.get_mut(&repo)?;
        record.record_act();
        record.chi = Some(new);
    }
    fed.save()?;

    match old {
        Some(prev) => {
            let delta = new - prev;
            let moved = delta.abs() > 1e-9;
            println!(
                "{repo}: χ {prev:.3} → {new:.3} ({}{:.3}). {}",
                if delta >= 0.0 { "+" } else { "" },
                delta,
                if moved {
                    "The repo's sense has MOVED."
                } else {
                    "The repo's sense is unchanged."
                }
            );
        }
        None => println!("{repo}: χ = {new:.3} (no prior value to compare)."),
    }
    Ok(())
}

fn cmd_run(repo: String, goal: String) -> Result<()> {
    let fed = Federation::discover(&cwd()?)?;
    let _record = fed.get(&repo)?; // validate the repo exists first
                                   // Execution organ (network-yield) — argv not yet pinned; degrade gracefully.
    eprintln!(
        "tracker: execution organ (network-yield) is not wired in yet.\n\
         Requested goal for {repo:?}: {goal:?}\n\
         Tracking and search are fully available; `run` will dispatch to the \
         network-yield CLI once its interface is pinned."
    );
    Err(TrackerError::ExecutionMissing)
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn ensure_purpose() -> Result<()> {
    if purpose::is_available() {
        Ok(())
    } else {
        Err(TrackerError::PurposeMissing(
            "`purpose --version` did not succeed".into(),
        ))
    }
}

fn print_salient(c: &chi::Character) {
    if c.salient.is_empty() {
        return;
    }
    println!("Load-bearing files (the sense surface):");
    for (name, deg) in c.salient.iter().take(6) {
        println!("  · {name}  (structural weight {deg:.1})");
    }
}

fn report_hits(repo: &str, q: &str, hits: &[purpose::AskHit]) {
    if hits.is_empty() {
        println!("{repo}: no matches for {q:?}.");
        return;
    }
    println!("{repo}: {} match(es) for {q:?} (freshly searched):", hits.len());
    for h in hits.iter().take(10) {
        println!("  {}:{}  [{}] {}", h.file, h.line, h.kind, h.name);
        if !h.snippet.is_empty() {
            println!("      {}", h.snippet);
        }
    }
}
