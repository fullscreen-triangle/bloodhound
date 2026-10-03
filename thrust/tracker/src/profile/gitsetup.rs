//! Point git at KeePassXC for every host in the profile, and give each host its
//! own commit identity.
//!
//! Credentials: for each host, the inherited helper list (e.g. Git Credential
//! Manager) is reset with an empty `helper` and replaced by `keepassxc`, which is
//! the `git-credential-keepassxc` binary talking to the running KeePassXC.
//!
//! Identity: for repos whose remotes live on one host, an `includeIf
//! hasconfig:remote.*.url:` rule sets that account's name and email. Canonical
//! repos managed by `tracker sync` are unaffected by this rule: their origins are
//! reached by URL, not configured as git remotes, and `tracker sync` maps the
//! identity per origin itself.

use super::{Account, Profile};
use crate::error::{Result, TrackerError};
use std::path::PathBuf;
use std::process::Command;

/// Only the Git group: elsewhere the helper may return a website login saved for
/// the same host (one imported from a browser, say), which forges reject.
const HELPER: &str = "keepassxc --git-groups";

fn helper_installed() -> bool {
    Command::new("git-credential-keepassxc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn identity_file(p: &Profile, a: &Account) -> PathBuf {
    let dir = p.path.parent().map(PathBuf::from).unwrap_or_default();
    dir.join(format!("identity-{}.gitconfig", a.id))
}

/// The helpers git would otherwise use for every host (e.g. Git Credential Manager).
fn inherited_helpers() -> Vec<String> {
    Command::new("git")
        .args(["config", "--get-all", "credential.helper"])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|h| !h.is_empty() && !h.starts_with("keepassxc"))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// `git config --global` arguments, one change each.
///
/// KeePassXC is asked first; the previously configured helpers stay behind it as a
/// fallback, so hosts whose token is not in KeePassXC yet keep working. With
/// `strict`, KeePassXC is the only helper for these hosts.
fn plan(p: &Profile, strict: bool) -> Vec<Vec<String>> {
    let fallback = if strict { Vec::new() } else { inherited_helpers() };
    let mut steps = Vec::new();
    for a in &p.accounts {
        let scope = format!("credential.https://{}", a.host);
        steps.push(vec!["--unset-all".into(), format!("{scope}.helper")]);
        steps.push(vec!["--add".into(), format!("{scope}.helper"), String::new()]);
        steps.push(vec!["--add".into(), format!("{scope}.helper"), HELPER.into()]);
        for h in &fallback {
            steps.push(vec!["--add".into(), format!("{scope}.helper"), h.clone()]);
        }
        // No `username` pin: the KeePassXC helper matches by URL, and a pin would hide
        // credentials the fallback stored under another name (e.g. GitLab's "oauth2").
        let file = identity_file(p, a).to_string_lossy().replace('\\', "/");
        for pattern in [format!("https://{}/**", a.host), format!("git@{}:**", a.host)] {
            steps.push(vec![format!("includeIf.hasconfig:remote.*.url:{pattern}.path"), file.clone()]);
        }
    }
    steps
}

fn show(step: &[String]) -> String {
    step.iter()
        .map(|s| if s.is_empty() || s.contains(' ') { format!("\"{s}\"") } else { s.clone() })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn run(p: &Profile, apply: bool, strict: bool) -> Result<()> {
    let steps = plan(p, strict);
    println!("Identity files (one per account):");
    for a in &p.accounts {
        println!("  {}  →  {} <{}>", identity_file(p, a).display(), a.name, a.email);
    }
    println!("\ngit config --global changes:");
    for s in &steps {
        println!("  git config --global {}", show(s));
    }

    if !apply {
        println!("\nNothing changed. Rerun with --apply to make these changes.");
        return Ok(());
    }

    for a in &p.accounts {
        let file = identity_file(p, a);
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(
            &file,
            format!(
                "# written by `tracker profile git-setup` for account {}\n[user]\n\tname = {}\n\temail = {}\n",
                a.id, a.name, a.email
            ),
        )?;
    }
    for s in &steps {
        let out = Command::new("git")
            .args(["config", "--global"])
            .args(s)
            .output()
            .map_err(|e| TrackerError::GitMissing(e.to_string()))?;
        // --unset-all exits 5 when there was nothing to unset.
        let benign = s[0] == "--unset-all" && out.status.code() == Some(5);
        if !out.status.success() && !benign {
            return Err(TrackerError::GitFailed {
                args: format!("config --global {}", show(s)),
                code: out.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            });
        }
    }
    println!("\nApplied.");

    println!("\nRemaining, in KeePassXC:");
    if !helper_installed() {
        println!("  · install the helper: cargo install git-credential-keepassxc");
    }
    println!("  · Settings → Browser Integration → enable it (no browser needs to be ticked)");
    println!("  · with the database unlocked, run once: git-credential-keepassxc configure");
    println!("  · one entry per host — URL https://<host>, username, and a personal access token as password:");
    for a in &p.accounts {
        println!("      https://{:<26} {}", a.host, a.user);
    }
    Ok(())
}
