//! The profile: one person across several forge accounts.
//!
//! It holds accounts and identities, never secrets. Credentials are asked of git's
//! credential protocol, which `tracker profile git-setup` points at KeePassXC, so
//! the password manager stays the only place a token lives.

pub(crate) mod forge;
mod gitsetup;

use crate::error::{Result, TrackerError};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const TEMPLATE: &str = r#"# tracker profile — you, across every forge you use.
# No secrets here: credentials come from KeePassXC through git's credential
# helper (see `tracker profile git-setup`).

[identity]                     # who you are on your canonical branches
name  = "Your Name"
email = "you@example.org"
# also = ["old@example.org"]   # further addresses that are also you

# One section per account. kind = github | gitlab | gitea.
# `name` / `email` are how that host sees your commits (default: [identity]).
[accounts.github]
kind = "github"
host = "github.com"
user = "you"
"#;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Forge {
    Github,
    Gitlab,
    Gitea,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    identity: RawIdentity,
    #[serde(default)]
    accounts: BTreeMap<String, RawAccount>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIdentity {
    name: String,
    email: String,
    #[serde(default)]
    also: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAccount {
    kind: Forge,
    host: String,
    user: String,
    name: Option<String>,
    email: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Account {
    pub id: String,
    pub kind: Forge,
    pub host: String,
    pub user: String,
    /// How this host sees your commits.
    pub name: String,
    pub email: String,
    /// True when no host-specific email was given and [identity]'s is used.
    pub email_is_default: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Profile {
    pub path: PathBuf,
    pub name: String,
    pub email: String,
    /// Every address that is you, lowercased.
    pub me: BTreeSet<String>,
    pub accounts: Vec<Account>,
}

impl Profile {
    /// `$TRACKER_PROFILE`, else `~/.tracker/profile.toml`.
    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os("TRACKER_PROFILE") {
            return PathBuf::from(p);
        }
        crate::registry::home_dir().join(".tracker").join("profile.toml")
    }

    /// The profile, or `None` if there is none yet.
    pub fn load() -> Result<Option<Profile>> {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text, path).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn require() -> Result<Profile> {
        Self::load()?.ok_or_else(|| {
            TrackerError::Profile(format!(
                "no profile at {}; create one with `tracker profile init`",
                Self::path().display()
            ))
        })
    }

    pub fn parse(text: &str, path: PathBuf) -> Result<Profile> {
        let raw: RawProfile = toml::from_str(text)
            .map_err(|e| TrackerError::Profile(format!("{}: {e}", path.display())))?;
        let mut me: BTreeSet<String> = std::iter::once(&raw.identity.email)
            .chain(&raw.identity.also)
            .map(|e| e.to_lowercase())
            .collect();
        let mut accounts = Vec::new();
        for (id, a) in raw.accounts {
            if let Some(e) = &a.email {
                me.insert(e.to_lowercase());
            }
            accounts.push(Account {
                name: a.name.unwrap_or_else(|| raw.identity.name.clone()),
                email_is_default: a.email.is_none(),
                email: a.email.unwrap_or_else(|| raw.identity.email.clone()),
                id,
                kind: a.kind,
                host: a.host.to_lowercase(),
                user: a.user,
            });
        }
        Ok(Profile {
            path,
            name: raw.identity.name,
            email: raw.identity.email,
            me,
            accounts,
        })
    }

    pub fn account(&self, id: &str) -> Option<&Account> {
        self.accounts.iter().find(|a| a.id == id)
    }

    /// The account whose host serves `url`, if exactly one does.
    pub fn account_for_url(&self, url: &str) -> Option<&Account> {
        let host = host_of(url)?;
        let mut hits = self.accounts.iter().filter(|a| a.host == host);
        let first = hits.next()?;
        hits.next().is_none().then_some(first)
    }
}

/// The host part of a git remote URL: `https://h/…`, `ssh://u@h:22/…`, `u@h:path`.
/// `None` for local paths.
pub fn host_of(url: &str) -> Option<String> {
    let host = if let Some((_, rest)) = url.split_once("://") {
        let authority = rest.split('/').next()?;
        let authority = authority.rsplit('@').next()?;
        authority.split(':').next()?
    } else {
        // scp-like `[user@]host:path`; a one-letter "host" is a Windows drive.
        let (before, _) = url.split_once(':')?;
        if before.contains('/') || before.contains('\\') {
            return None;
        }
        before.rsplit('@').next()?
    };
    (host.len() > 1).then(|| host.to_lowercase())
}

// ── commands ─────────────────────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum ProfileCmd {
    /// Write a template profile (to $TRACKER_PROFILE or ~/.tracker/profile.toml).
    Init,

    /// Show the profile: identities, accounts, and how each host sees you.
    Show,

    /// List your repositories on every account and group copies of the same project.
    Repos {
        /// Only these accounts (repeatable).
        #[arg(long = "account")]
        accounts: Vec<String>,
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },

    /// Point git at KeePassXC for these hosts, and give each host its own identity.
    ///
    /// Prints the git config changes; `--apply` makes them.
    GitSetup {
        #[arg(long)]
        apply: bool,
        /// Make KeePassXC the only credential helper for these hosts (default: it is
        /// asked first, and the previous helper stays behind it as a fallback).
        #[arg(long)]
        strict: bool,
    },
}

pub fn run(cmd: ProfileCmd) -> Result<()> {
    match cmd {
        ProfileCmd::Init => cmd_init(),
        ProfileCmd::Show => cmd_show(),
        ProfileCmd::Repos { accounts, json } => cmd_repos(accounts, json),
        ProfileCmd::GitSetup { apply, strict } => gitsetup::run(&Profile::require()?, apply, strict),
    }
}

fn cmd_init() -> Result<()> {
    let path = Profile::path();
    if path.exists() {
        return Err(TrackerError::Profile(format!("{} already exists", path.display())));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, TEMPLATE)?;
    println!("Wrote {}. Fill in your accounts, then `tracker profile show`.", path.display());
    Ok(())
}

fn cmd_show() -> Result<()> {
    let p = Profile::require()?;
    println!("Profile: {}", p.path.display());
    println!("You: {} <{}>", p.name, p.email);
    let others: Vec<&String> = p.me.iter().filter(|e| **e != p.email.to_lowercase()).collect();
    if !others.is_empty() {
        println!(
            "Also you: {}",
            others.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
    println!();
    println!("{:<12} {:<7} {:<26} {:<22} SEEN AS", "ACCOUNT", "KIND", "HOST", "USER");
    for a in &p.accounts {
        let kind = match a.kind {
            Forge::Github => "github",
            Forge::Gitlab => "gitlab",
            Forge::Gitea => "gitea",
        };
        println!(
            "{:<12} {:<7} {:<26} {:<22} {} <{}>{}",
            a.id,
            kind,
            a.host,
            a.user,
            a.name,
            a.email,
            if a.email_is_default { "  (no host email set; using yours)" } else { "" }
        );
    }
    Ok(())
}

/// Every repository on the chosen accounts, copies of one project grouped together.
#[derive(Serialize)]
pub struct Repos {
    pub accounts: Vec<String>,
    pub projects: Vec<forge::Project>,
    /// Accounts that could only be listed partly, or not at all, and why.
    pub notes: Vec<String>,
}

pub fn repos(only: &[String]) -> Result<Repos> {
    let p = Profile::require()?;
    let accounts: Vec<&Account> = if only.is_empty() {
        p.accounts.iter().collect()
    } else {
        only.iter()
            .map(|id| {
                p.account(id)
                    .ok_or_else(|| TrackerError::Profile(format!("no account {id:?} in the profile")))
            })
            .collect::<Result<_>>()?
    };

    let mut all = Vec::new();
    let mut notes = Vec::new();
    for a in &accounts {
        match forge::list(a) {
            Ok(listing) => {
                if listing.refused {
                    notes.push(format!(
                        "{}: {} refused the credential git supplied (a login password, not an API token?) —                          public repositories only. Store a personal access token for it in KeePassXC.",
                        a.id, a.host
                    ));
                } else if !listing.authenticated {
                    notes.push(format!(
                        "{}: no credential from git for {} — public repositories only",
                        a.id, a.host
                    ));
                }
                all.extend(listing.repos);
            }
            Err(e) => notes.push(format!("{}: {e}", a.id)),
        }
    }
    Ok(Repos {
        accounts: accounts.iter().map(|a| a.id.clone()).collect(),
        projects: forge::group(all),
        notes,
    })
}

fn cmd_repos(only: Vec<String>, json: bool) -> Result<()> {
    let r = repos(&only)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r.projects).expect("serialisable"));
    } else {
        print_groups(&r.projects, &r.accounts);
    }
    for n in &r.notes {
        eprintln!("note: {n}");
    }
    Ok(())
}

fn print_groups(groups: &[forge::Project], accounts: &[String]) {
    let width = groups
        .iter()
        .map(|g| g.project.len())
        .max()
        .unwrap_or(7)
        .clamp(7, 40);
    print!("{:<width$}", "PROJECT");
    for a in accounts {
        print!("  {a:^10}");
    }
    println!();
    for g in groups {
        print!("{:<width$}", g.project);
        for a in accounts {
            let cell = match g.copies.iter().find(|c| c.account == *a) {
                Some(c) if c.private => "private",
                Some(_) => "public",
                None => "·",
            };
            print!("  {cell:^10}");
        }
        println!();
    }
    let shared = groups.iter().filter(|g| g.copies.len() > 1).count();
    println!(
        "\n{} project(s); {} live on more than one host.",
        groups.len(),
        shared
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_from_every_url_shape() {
        assert_eq!(host_of("https://gitlab.com/a/b.git").as_deref(), Some("gitlab.com"));
        assert_eq!(host_of("https://user@Git.Uni-Greifswald.de/a").as_deref(), Some("git.uni-greifswald.de"));
        assert_eq!(host_of("ssh://git@gitlab.bitspark.com:2222/x/y").as_deref(), Some("gitlab.bitspark.com"));
        assert_eq!(host_of("git@github.com:fullscreen-triangle/bloodhound.git").as_deref(), Some("github.com"));
        assert_eq!(host_of("C:/Users/me/repo.git"), None);
        assert_eq!(host_of("/srv/repo.git"), None);
        assert_eq!(host_of("../relative/repo"), None);
    }

    #[test]
    fn every_account_address_is_you_and_defaults_are_marked() {
        let p = Profile::parse(
            r#"
            [identity]
            name = "K"
            email = "K@Home.org"
            also = ["old@x.org"]
            [accounts.co]
            kind = "gitlab"
            host = "GitLab.Co.com"
            user = "k"
            email = "k@co.com"
            [accounts.gh]
            kind = "github"
            host = "github.com"
            user = "kgh"
            "#,
            PathBuf::from("p.toml"),
        )
        .unwrap();
        for addr in ["k@home.org", "old@x.org", "k@co.com"] {
            assert!(p.me.contains(addr), "{addr}");
        }
        assert!(!p.me.contains("boss@co.com"));
        assert_eq!(p.account_for_url("https://gitlab.co.com/g/r.git").unwrap().id, "co");
        let gh = p.account("gh").unwrap();
        assert!(gh.email_is_default);
        assert_eq!(gh.email, "K@Home.org");
    }

    #[test]
    fn template_parses() {
        Profile::parse(TEMPLATE, PathBuf::from("t")).unwrap();
    }
}
