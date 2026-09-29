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

    /// Keep a repo's several origins (GitHub, GitLab, company/university GitLab) in step,
    /// each seeing only the paths its visibility manifest allows.
    Sync {
        #[command(subcommand)]
        action: crate::sync::SyncCmd,
    },

    /// You across every forge: accounts, per-host identities, and all your repos in one view.
    /// Holds no secrets; credentials come from KeePassXC through git.
    Profile {
        #[command(subcommand)]
        action: crate::profile::ProfileCmd,
    },

    /// Keep every account's token alive: check them, and renew them before they expire.
    /// Tokens live only in KeePassXC; tracker keeps just their expiry dates.
    Tokens {
        #[command(subcommand)]
        action: TokensCmd,
    },

    /// List every question and action this tool offers, with its JSON argument schema.
    Describe,

    /// Run one operation with JSON arguments and print its JSON result.
    ///
    /// Failures are printed as {"error": {"code", "message"}} and exit non-zero.
    /// Example: tracker call sync_visibility '{"paths": ["bitspark/pricing.md"]}'
    Call {
        /// Operation name (see `tracker describe`).
        op: String,
        /// Arguments as a JSON object; `-` reads them from stdin. Default: {}.
        args: Option<String>,
    },

    /// Serve every operation to AI agents over the Model Context Protocol (stdio).
    Mcp {
        /// Directory to treat as the current repo (default: where the server starts).
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum TokensCmd {
    /// Check every token: working, expiring, missing or rejected. Renews nothing.
    Status {
        #[arg(long = "account")]
        accounts: Vec<String>,
    },
    /// Check every token and renew the ones that are due (GitLab rotates itself).
    Refresh {
        #[arg(long = "account")]
        accounts: Vec<String>,
        /// Renew GitLab tokens now even if they are not due.
        #[arg(long)]
        force: bool,
        /// For scheduled runs: never wait for KeePassXC to be unlocked; log to ~/.tracker/tokens.log.
        #[arg(long)]
        unattended: bool,
    },
    /// Store a token you just made for an account (checked against the forge first).
    ///
    /// Opens the page where the token is made, then reads it from stdin.
    Set {
        account: String,
        /// Do not open the token page in the browser.
        #[arg(long)]
        no_open: bool,
    },
    /// Renew tokens daily in the background (Windows Task Scheduler; prints a cron line elsewhere).
    Schedule {
        /// Remove the daily renewal instead.
        #[arg(long)]
        off: bool,
    },
}

impl Cli {
    /// True for commands whose output is read by programs, so errors go to stdout as JSON.
    pub fn machine(&self) -> bool {
        matches!(self.command, Cmd::Describe | Cmd::Call { .. })
    }
}

/// Entry point from `main`.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Cmd::Tokens { action } => cmd_tokens(action),
        Cmd::Describe => {
            println!("{}", pretty(&crate::api::describe()));
            Ok(())
        }
        Cmd::Call { op, args } => {
            let text = match args.as_deref() {
                None => "{}".to_string(),
                Some("-") => std::io::read_to_string(std::io::stdin())?,
                Some(s) => s.to_string(),
            };
            let args: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| TrackerError::BadArgs(format!("arguments are not JSON: {e}")))?;
            println!("{}", pretty(&crate::api::call(&op, args)?));
            Ok(())
        }
        Cmd::Mcp { root } => {
            if let Some(r) = root {
                std::env::set_current_dir(&r).map_err(|_| TrackerError::BadRepoPath(r))?;
            }
            Ok(crate::mcp::serve()?)
        }
        Cmd::Init => cmd_init(),
        Cmd::Add { path, name, remote } => cmd_add(path, name, remote),
        Cmd::List => cmd_list(),
        Cmd::Sense { repo, question } => cmd_sense(repo, question),
        Cmd::Ask { question } => cmd_ask(question),
        Cmd::Drift { repo } => cmd_drift(repo),
        Cmd::Run { repo, goal } => cmd_run(repo, goal),
        Cmd::Sync { action } => crate::sync::run(action),
        Cmd::Profile { action } => crate::profile::run(action),
    }
}

fn cwd() -> Result<PathBuf> {
    Ok(std::env::current_dir()?)
}

fn pretty(v: &serde_json::Value) -> String {
    serde_json::to_string_pretty(v).expect("JSON values serialise")
}

fn cmd_init() -> Result<()> {
    let root = cwd()?;
    Federation::init(&root)?;
    println!("Initialised federation at {}", root.display());
    println!("Next: `tracker add <path-to-repo>`");
    Ok(())
}

fn cmd_add(path: PathBuf, name: Option<String>, remote: Option<String>) -> Result<()> {
    let mut fed = Federation::locate(&cwd()?)?;
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
        "Added {name}: χ = {:.3} (core {} of {} blocks; {} fragment(s)) (m = 1).",
        character.chi, character.core_blocks, character.blocks, character.fragments
    );
    print_salient(&character);
    Ok(())
}

fn cmd_list() -> Result<()> {
    let fed = Federation::locate(&cwd()?)?;
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
    let fed = Federation::locate(&cwd()?)?;
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
                "{repo}: sense/goal — χ = {:.3} (core {} of {} blocks; {} fragment(s)).",
                character.chi, character.core_blocks, character.blocks, character.fragments
            );
            print_salient(&character);
            println!(
                "\nCheapest conceptual split of the core (χ severs this region of {} block(s)):",
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
    let fed = Federation::locate(&cwd()?)?;
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
    // Reconstruct the current sense (construction phase, I4), then commit it (I2).
    let d = crate::api::drift(&repo)?;
    let new = d.current;
    match d.previous {
        Some(prev) => {
            let delta = new - prev;
            println!(
                "{repo}: χ {prev:.3} → {new:.3} ({}{:.3}). {}",
                if delta >= 0.0 { "+" } else { "" },
                delta,
                if d.moved {
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
    let fed = Federation::locate(&cwd()?)?;
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

fn print_token_statuses(statuses: &[crate::tokens::Status]) {
    println!("{:<10} {:<24} {:<10} {:<18} ACTION", "ACCOUNT", "HOST", "STATE", "EXPIRES");
    for s in statuses {
        let state = serde_json::to_value(s.state)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let expires = match (&s.expires_at, s.days_left) {
            (Some(e), Some(d)) => format!("{e} ({d}d)"),
            _ if matches!(s.state, crate::tokens::State::Ok | crate::tokens::State::Rotated) => "never".to_string(),
            _ => "—".to_string(),
        };
        println!(
            "{:<10} {:<24} {:<10} {:<18} {}",
            s.account,
            s.host,
            state,
            expires,
            s.action.as_deref().unwrap_or("")
        );
        if let Some(u) = &s.issue_url {
            println!("{:<10} {u}", "");
        }
    }
}

fn open_in_browser(url: &str) {
    let _ = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).status()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).status()
    } else {
        std::process::Command::new("xdg-open").arg(url).status()
    };
}

fn cmd_tokens(cmd: TokensCmd) -> Result<()> {
    use crate::tokens::{self, Options};
    match cmd {
        TokensCmd::Status { accounts } => {
            let o = Options { rotate: false, force: false, wait_unlock: true };
            print_token_statuses(&tokens::refresh(&accounts, &o)?);
        }
        TokensCmd::Refresh { accounts, force, unattended } => {
            let o = Options { rotate: true, force, wait_unlock: !unattended };
            let result = tokens::refresh(&accounts, &o);
            if unattended {
                let log = crate::profile::Profile::path().with_file_name("tokens.log");
                let line = match &result {
                    Ok(s) => serde_json::to_string(s).unwrap_or_default(),
                    Err(e) => format!("error: {e}"),
                };
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
                    use std::io::Write as _;
                    let _ = writeln!(f, "{stamp} {line}");
                }
            }
            print_token_statuses(&result?);
        }
        TokensCmd::Set { account, no_open } => {
            let profile = crate::profile::Profile::require()?;
            let a = profile
                .account(&account)
                .ok_or_else(|| TrackerError::Profile(format!("no account {account:?} in the profile")))?;
            let url = tokens::issue_url(a);
            if !no_open {
                open_in_browser(&url);
            }
            eprintln!("Make a token for {} at:\n  {url}", a.host);
            eprintln!("Paste it here and press Enter (it goes straight to KeePassXC):");
            let mut token = String::new();
            std::io::stdin().read_line(&mut token)?;
            print_token_statuses(&[tokens::set(&account, &token)?]);
        }
        TokensCmd::Schedule { off } => println!("{}", tokens::schedule(!off)?),
    }
    Ok(())
}
