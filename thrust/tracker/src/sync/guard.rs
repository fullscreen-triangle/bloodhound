//! The pre-push guard: the canonical branch carries hidden work, so a plain
//! `git push` of it to any remote in the manifest would leak. The hook refuses
//! such pushes unless they come from `tracker sync` itself.

use super::git::Git;
use super::manifest::Manifest;
use crate::error::Result;
use std::path::PathBuf;

/// Set by `tracker sync` on its own pushes; the hook lets those through.
pub const PUSH_ENV: &str = "TRACKER_SYNC_PUSH";
const MARKER: &str = "# tracker-sync guard";

#[derive(Debug, serde::Serialize)]
#[serde(tag = "state", content = "path", rename_all = "snake_case")]
pub enum HookStatus {
    Installed,
    Unchanged,
    /// A pre-push hook not written by tracker exists; it was left alone.
    Foreign(PathBuf),
}

fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn script(manifest: &Manifest) -> String {
    let mut urls: Vec<String> = Vec::new();
    for r in &manifest.remotes {
        let base = r.url.trim_end_matches('/');
        let alt = match base.strip_suffix(".git") {
            Some(stripped) => stripped.to_string(),
            None => format!("{base}.git"),
        };
        for u in [r.url.clone(), base.to_string(), alt] {
            if !urls.contains(&u) {
                urls.push(u);
            }
        }
    }
    let mut s = format!(
        "#!/bin/sh\n{MARKER} (generated; refreshed on every `tracker sync run`)\n\
         [ \"${PUSH_ENV}\" = \"1\" ] && exit 0\n"
    );
    if !urls.is_empty() {
        let pattern = urls.iter().map(|u| quote(u)).collect::<Vec<_>>().join("|");
        s.push_str(&format!(
            "case \"$2\" in\n  {pattern})\n    echo \"tracker: refusing a direct push to $2 — this branch carries \
             work hidden from that remote. Use 'tracker sync run'.\" >&2\n    exit 1 ;;\nesac\n"
        ));
    }
    s.push_str("exit 0\n");
    s
}

pub fn install(git: &Git, manifest: &Manifest) -> Result<HookStatus> {
    let path = git.git_path("hooks/pre-push")?;
    let wanted = script(manifest);
    match std::fs::read_to_string(&path) {
        Ok(existing) if existing == wanted => return Ok(HookStatus::Unchanged),
        Ok(existing) if !existing.contains(MARKER) => return Ok(HookStatus::Foreign(path)),
        _ => {}
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, wanted)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(HookStatus::Installed)
}
