//! Changing the visibility manifest on request, keeping its comments and layout.
//!
//! The copy committed on the canonical branch is the one `sync` obeys, so an edit is
//! committed there by default (only `.sync.toml`; nothing else staged is touched).

use super::engine;
use super::git::Git;
use super::manifest::{self, Manifest, MixedMessages};
use crate::error::{Result, TrackerError};
use serde::Serialize;
use toml_edit::{value, Array, DocumentMut};

#[derive(Debug, Serialize)]
pub struct Edited {
    /// False when the manifest already said this.
    pub changed: bool,
    /// The commit that records the change, when one was made.
    pub commit: Option<String>,
}

fn normalise(s: &str) -> String {
    s.replace("\r\n", "\n")
}

fn edit(git: &Git, summary: &str, commit: bool, f: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<Edited> {
    let current = engine::load_manifest(git)?;
    let branch_ref = format!("refs/heads/{}", current.branch);
    let committed = git
        .show_file(&branch_ref, manifest::FILE)?
        .ok_or_else(|| TrackerError::Internal("manifest vanished after loading".into()))?;
    let path = git.root().join(manifest::FILE);
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|_| committed.clone());

    if commit {
        if git.head_branch()?.as_deref() != Some(branch_ref.as_str()) {
            return Err(TrackerError::Manifest(format!(
                "check out the canonical branch {:?} to change the manifest",
                current.branch
            )));
        }
        if normalise(&on_disk) != normalise(&committed) {
            return Err(TrackerError::Manifest(format!(
                "{} has uncommitted edits; commit or discard them first",
                manifest::FILE
            )));
        }
    }

    let mut doc: DocumentMut = on_disk
        .parse()
        .map_err(|e| TrackerError::Manifest(format!("{}: {e}", manifest::FILE)))?;
    f(&mut doc)?;
    let updated = doc.to_string();
    // Refuse to write anything `sync` could not then read.
    Manifest::parse(&updated)?;
    if normalise(&updated) == normalise(&on_disk) {
        return Ok(Edited {
            changed: false,
            commit: None,
        });
    }
    std::fs::write(&path, &updated)?;
    if !commit {
        return Ok(Edited {
            changed: true,
            commit: None,
        });
    }
    let message = format!("tracker: {summary}");
    git.run(&["commit", "-q", "-m", &message, "--", manifest::FILE])?;
    Ok(Edited {
        changed: true,
        commit: git.rev("HEAD")?,
    })
}

/// Hide paths matching `pattern` from every remote that does not see `label`.
pub fn hide(git: &Git, pattern: &str, label: &str, commit: bool) -> Result<Edited> {
    edit(git, &format!("hide {pattern} as {label}"), commit, |doc| {
        doc["hide"][pattern] = value(label);
        Ok(())
    })
}

/// Drop the hide rule for exactly `pattern`.
pub fn unhide(git: &Git, pattern: &str, commit: bool) -> Result<Edited> {
    edit(git, &format!("stop hiding {pattern}"), commit, |doc| {
        let removed = doc
            .get_mut("hide")
            .and_then(|h| h.as_table_like_mut())
            .and_then(|t| t.remove(pattern));
        match removed {
            Some(_) => Ok(()),
            None => Err(TrackerError::Manifest(format!("no hide rule for {pattern:?}"))),
        }
    })
}

pub struct RemoteSpec<'a> {
    pub name: &'a str,
    pub url: &'a str,
    pub sees: &'a [String],
    pub branch: Option<&'a str>,
    pub account: Option<&'a str>,
    pub mixed_messages: Option<MixedMessages>,
}

/// Add a remote, or replace the given settings of an existing one.
pub fn set_remote(git: &Git, r: &RemoteSpec, commit: bool) -> Result<Edited> {
    edit(git, &format!("set remote {}", r.name), commit, |doc| {
        let t = &mut doc["remotes"][r.name];
        t["url"] = value(r.url);
        t["sees"] = value(r.sees.iter().collect::<Array>());
        if let Some(b) = r.branch {
            t["branch"] = value(b);
        }
        if let Some(a) = r.account {
            t["account"] = value(a);
        }
        if let Some(m) = r.mixed_messages {
            let m = match m {
                MixedMessages::Refuse => "refuse",
                MixedMessages::Generic => "generic",
                MixedMessages::Keep => "keep",
            };
            t["mixed_messages"] = value(m);
        }
        Ok(())
    })
}
