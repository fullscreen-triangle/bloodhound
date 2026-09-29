//! Hand-resolution of a remote commit the engine could not replay by itself
//! (a conflict, or a change to a path the manifest hides from that remote).
//!
//! The commit is applied to the working tree with `cherry-pick --no-commit` — so
//! nothing of the remote's own history enters the canonical branch — and, once the
//! human has fixed it, committed with the collaborator's identities and a
//! `Sync-Origin` trailer, exactly as an automatic replay would have been.

use super::engine::{self, Options, RESOLVE_STATE};
use super::git::Git;
use crate::error::{Result, TrackerError};

#[derive(serde::Serialize)]
pub struct Started {
    pub remote: String,
    pub commit: String,
    pub subject: String,
    pub conflicts: Vec<String>,
}

fn unmerged(git: &Git) -> Result<Vec<String>> {
    Ok(git
        .run(&["diff", "--name-only", "--diff-filter=U"])?
        .lines()
        .map(str::to_string)
        .collect())
}

fn read_state(git: &Git) -> Result<(String, String)> {
    let text = std::fs::read_to_string(git.sync_dir().join(RESOLVE_STATE))
        .map_err(|_| TrackerError::Resolve("no `tracker sync resolve` is in progress".into()))?;
    let mut it = text.split_whitespace();
    match (it.next(), it.next()) {
        (Some(r), Some(s)) => Ok((r.to_string(), s.to_string())),
        _ => Err(TrackerError::Resolve("resolve state file is corrupt".into())),
    }
}

fn cleanup(git: &Git) -> Result<()> {
    let _ = std::fs::remove_file(git.sync_dir().join(RESOLVE_STATE));
    let _ = git.raw(&["cherry-pick", "--quit"], &[], None);
    let _ = std::fs::remove_file(git.git_path("MERGE_MSG")?);
    Ok(())
}

pub fn start(git: &Git, remote: Option<&str>) -> Result<Started> {
    if git.sync_dir().join(RESOLVE_STATE).exists() {
        return Err(TrackerError::ResolveInProgress);
    }
    let manifest = engine::load_manifest(git)?;
    let branch_ref = format!("refs/heads/{}", manifest.branch);
    if git.head_branch()?.as_deref() != Some(branch_ref.as_str()) {
        return Err(TrackerError::Resolve(format!(
            "check out the canonical branch {:?} first",
            manifest.branch
        )));
    }
    if git.is_dirty()? {
        return Err(TrackerError::DirtyWorktree);
    }

    // Replay everything up to the blocked commit, and learn which commit that is.
    let opts = Options {
        dry_run: false,
        pull: true,
        push: false,
        only: remote.map(|r| vec![r.to_string()]).unwrap_or_default(),
        trust_messages: false,
    };
    let (remote, commit) = match engine::sync(git, &opts) {
        Ok(_) => {
            return Err(TrackerError::Resolve(
                "nothing is blocked; `tracker sync run` goes through as is".into(),
            ))
        }
        Err(TrackerError::SyncConflict { remote, commit, .. })
        | Err(TrackerError::HiddenFromRemote { remote, commit, .. }) => (remote, commit),
        Err(other) => return Err(other),
    };

    let rc = git.commit(&commit)?;
    let mut args = vec!["cherry-pick", "--no-commit"];
    if rc.parents.len() > 1 {
        args.extend(["-m", "1"]);
    }
    args.push(&commit);
    let out = git.raw(&args, &[], None)?;
    if out.code > 1 {
        return Err(TrackerError::GitFailed {
            args: args.join(" "),
            code: out.code,
            stderr: out.stderr,
        });
    }
    std::fs::create_dir_all(git.sync_dir())?;
    std::fs::write(git.sync_dir().join(RESOLVE_STATE), format!("{remote} {commit}\n"))?;

    Ok(Started {
        remote,
        subject: rc.subject().to_string(),
        commit,
        conflicts: unmerged(git)?,
    })
}

/// Commit the resolution; returns the new canonical commit.
pub fn finish(git: &Git) -> Result<String> {
    let (remote, commit) = read_state(git)?;
    let still = unmerged(git)?;
    if !still.is_empty() {
        return Err(TrackerError::Resolve(format!(
            "still unmerged: {} — fix them and `git add` them first",
            still.join(", ")
        )));
    }
    let tree = git.run(&["write-tree"])?.trim().to_string();
    let head = git
        .rev("HEAD")?
        .ok_or_else(|| TrackerError::Resolve("HEAD does not resolve".into()))?;
    let rc = git.commit(&commit)?;
    let message = engine::with_origin(&rc.message, &remote, &commit);
    let resolved = git.commit_tree(&tree, std::slice::from_ref(&head), &rc.author, &rc.committer, &message)?;
    git.update_ref("HEAD", &resolved, Some(&head))?;
    cleanup(git)?;
    Ok(resolved)
}

pub fn abort(git: &Git) -> Result<()> {
    read_state(git)?;
    git.run(&["reset", "--merge"])?;
    cleanup(git)
}
