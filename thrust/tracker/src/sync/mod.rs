//! `tracker sync` — one repo, several origins, each seeing only what it may.
//!
//! The canonical branch holds everything and is never pushed as-is to any origin
//! in the manifest. Each origin (GitHub, gitlab.com, a company or university
//! GitLab) receives a *projection* of it: the same history with the paths hidden
//! from that origin filtered out. Collaborators' commits on any origin are pulled
//! back into the canonical branch and flow on to every other origin that may see
//! what they touched.

pub(crate) mod edit;
pub(crate) mod engine;
pub(crate) mod git;
pub(crate) mod guard;
pub(crate) mod manifest;
pub(crate) mod resolve;

use crate::error::{Result, TrackerError};
use crate::registry::Federation;
use clap::Subcommand;
use engine::{short, Options, Outcome, Report};
use git::Git;
use guard::HookStatus;

#[derive(Subcommand)]
pub enum SyncCmd {
    /// Write a template `.sync.toml` here; edit it, then commit it on the canonical branch.
    Init {
        /// Tracked repo name (default: the repo containing the current directory).
        #[arg(long)]
        repo: Option<String>,
    },

    /// Show what a sync would do. Fetches, but changes and pushes nothing.
    Status {
        #[arg(long)]
        repo: Option<String>,
        /// Only these remotes (repeatable).
        #[arg(long = "remote")]
        remotes: Vec<String>,
    },

    /// Pull collaborators' commits into the canonical branch, then push every remote its projection.
    Run {
        #[arg(long)]
        repo: Option<String>,
        /// Only these remotes (repeatable).
        #[arg(long = "remote")]
        remotes: Vec<String>,
        /// Do not pull collaborators' commits.
        #[arg(long)]
        no_pull: bool,
        /// Do not push projections.
        #[arg(long)]
        no_push: bool,
        /// Keep the original message of commits that also touched hidden paths.
        #[arg(long)]
        trust_messages: bool,
    },

    /// Set the message a remote sees for a canonical commit that also touched paths hidden from it.
    ///
    /// Stored as a git note (refs/notes/sync-messages), so history is not rewritten.
    Message {
        /// The canonical commit.
        commit: String,
        /// The one-line message the remote(s) should see.
        text: String,
        /// The remote this message is for.
        #[arg(long, conflicts_with = "shared")]
        remote: Option<String>,
        /// Use this message for every remote without a message of its own.
        #[arg(long)]
        shared: bool,
        #[arg(long)]
        repo: Option<String>,
    },

    /// Apply the next blocked remote commit to the working tree so it can be fixed by hand.
    Resolve {
        /// The remote whose commit is blocked (default: the first one found).
        remote: Option<String>,
        /// Commit the fixed-up resolution.
        #[arg(long = "continue", conflicts_with = "abort")]
        cont: bool,
        /// Abandon the resolution and restore the working tree.
        #[arg(long)]
        abort: bool,
        #[arg(long)]
        repo: Option<String>,
    },
}

pub fn run(cmd: SyncCmd) -> Result<()> {
    match cmd {
        SyncCmd::Init { repo } => cmd_init(repo),
        SyncCmd::Status { repo, remotes } => {
            let git = open(repo)?;
            let report = engine::sync(
                &git,
                &Options {
                    dry_run: true,
                    pull: true,
                    push: true,
                    only: remotes,
                    trust_messages: false,
                },
            )?;
            print_report(&report, true);
            Ok(())
        }
        SyncCmd::Run {
            repo,
            remotes,
            no_pull,
            no_push,
            trust_messages,
        } => {
            let git = open(repo)?;
            let report = engine::sync(
                &git,
                &Options {
                    dry_run: false,
                    pull: !no_pull,
                    push: !no_push,
                    only: remotes,
                    trust_messages,
                },
            )?;
            print_report(&report, false);
            Ok(())
        }
        SyncCmd::Message {
            commit,
            text,
            remote,
            shared,
            repo,
        } => cmd_message(open(repo)?, commit, text, remote, shared),
        SyncCmd::Resolve {
            remote,
            cont,
            abort,
            repo,
        } => cmd_resolve(open(repo)?, remote, cont, abort),
    }
}

fn open(repo: Option<String>) -> Result<Git> {
    let cwd = std::env::current_dir()?;
    let path = match repo {
        Some(name) => Federation::locate(&cwd)?.get(&name)?.path.clone(),
        None => cwd,
    };
    Git::open(&path)
}

fn cmd_init(repo: Option<String>) -> Result<()> {
    let git = open(repo)?;
    let path = git.root().join(manifest::FILE);
    if path.exists() {
        return Err(TrackerError::Manifest(format!("{} already exists", path.display())));
    }
    std::fs::write(&path, manifest::TEMPLATE)?;
    println!("Wrote {}.", path.display());
    println!("Edit the remotes and hidden paths, commit it on the canonical branch, then `tracker sync status`.");
    Ok(())
}

/// Set the message `remote` (or, with `shared`, every remote without its own) sees
/// for a canonical commit. Returns the full commit sha.
pub fn set_message(git: &Git, commit: &str, text: &str, remote: Option<&str>, shared: bool) -> Result<String> {
    let manifest = engine::load_manifest(git)?;
    let key = match (remote, shared) {
        (Some(r), false) => {
            manifest.remote(r)?;
            engine::message_key(r)
        }
        (None, true) => engine::SHARED_MESSAGE.to_string(),
        _ => {
            return Err(TrackerError::Manifest(
                "give exactly one of --remote <name> or --shared".into(),
            ))
        }
    };
    let text = text.trim();
    if text.is_empty() || text.contains('\n') {
        return Err(TrackerError::Manifest("the message must be a single non-empty line".into()));
    }
    let sha = git
        .rev(&commit)?
        .ok_or_else(|| TrackerError::Resolve(format!("no such commit: {commit}")))?;
    let prefix = format!("{key}: ");
    let mut lines: Vec<String> = git
        .note(engine::NOTES_REF, &sha)?
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with(&prefix))
        .map(str::to_string)
        .collect();
    lines.push(format!("{key}: {text}"));
    git.set_note(engine::NOTES_REF, &sha, &(lines.join("\n") + "\n"))?;
    Ok(sha)
}

fn cmd_message(git: Git, commit: String, text: String, remote: Option<String>, shared: bool) -> Result<()> {
    let sha = set_message(&git, &commit, &text, remote.as_deref(), shared)?;
    let text = text.trim();
    let who = remote.unwrap_or_else(|| "every remote without its own message".into());
    println!("{} will appear to {who} as: {text}", short(&sha));
    println!("(Stored in refs/notes/{}; push that ref to your private canonical remote to keep it.)", engine::NOTES_REF);
    Ok(())
}

fn cmd_resolve(git: Git, remote: Option<String>, cont: bool, abort: bool) -> Result<()> {
    if abort {
        resolve::abort(&git)?;
        println!("Resolution abandoned; the working tree is back as it was.");
        return Ok(());
    }
    if cont {
        let c = resolve::finish(&git)?;
        println!("Committed the resolution as {}. Run `tracker sync run` to carry on.", short(&c));
        return Ok(());
    }
    let s = resolve::start(&git, remote.as_deref())?;
    println!(
        "Applied {} commit {} ({:?}) to the working tree, uncommitted.",
        s.remote,
        short(&s.commit),
        s.subject
    );
    if s.conflicts.is_empty() {
        println!("It applied cleanly. Review or edit it, `git add` your changes, then `tracker sync resolve --continue`.");
    } else {
        println!("Conflicts in:");
        for p in &s.conflicts {
            println!("  {p}");
        }
        println!("Fix them, `git add` them, then `tracker sync resolve --continue` (or --abort).");
    }
    Ok(())
}

fn print_report(report: &Report, dry: bool) {
    if let Some(HookStatus::Foreign(path)) = &report.hook {
        eprintln!(
            "warning: {} is not tracker's own hook, so direct pushes of the canonical branch to \
             these remotes are NOT blocked. See the README to combine the two.",
            path.display()
        );
    }
    if let Some((from, to)) = &report.advanced {
        println!(
            "{}: {} {} → {}",
            report.branch,
            if dry { "would advance" } else { "advanced" },
            short(from),
            short(to)
        );
    }
    for r in &report.remotes {
        let mut notes = Vec::new();
        if !r.replayed.is_empty() {
            notes.push(format!("{} collaborator commit(s) pulled", r.replayed.len()));
        }
        if r.adopted > 0 {
            notes.push(format!("{} earlier projection(s) re-recognised", r.adopted));
        }
        if r.created > 0 {
            notes.push(format!("{} new commit(s)", r.created));
        }
        if r.dropped > 0 {
            notes.push(format!("{} hidden-only commit(s) left out", r.dropped));
        }
        let status = match &r.outcome {
            None => "not pushed".to_string(),
            Some(Outcome::UpToDate) => "up to date".to_string(),
            Some(Outcome::NothingVisible) => "nothing visible to it yet".to_string(),
            Some(Outcome::Pushed { from, to }) => format!(
                "pushed {}..{}",
                from.as_deref().map(short).unwrap_or("(new branch)"),
                short(to)
            ),
            Some(Outcome::WouldPush { from, to }) => format!(
                "would push {}..{}",
                from.as_deref().map(short).unwrap_or("(new branch)"),
                short(to)
            ),
        };
        let notes = if notes.is_empty() {
            String::new()
        } else {
            format!("  ({})", notes.join(", "))
        };
        println!("  {:<14} {status}{notes}", r.name);
    }
}
