//! Git on a user's repos, as tracker operations: a repo's state, read-only git
//! commands, any git command (an action, so a person confirms it), pushing a branch
//! to the origin a profile account owns, and opening a GitHub Codespace.

use crate::error::{Result, TrackerError};
use crate::profile::{host_of, Forge, Profile};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Result<(bool, String, String)> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|e| TrackerError::GitMissing(e.to_string()))?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn git_ok(dir: &Path, args: &[&str]) -> Result<String> {
    let (ok, out, err) = git(dir, args)?;
    if !ok {
        return Err(TrackerError::GitFailed {
            args: args.join(" "),
            code: 1,
            stderr: err.trim().to_string(),
        });
    }
    Ok(out)
}

/// Branch, remotes, working-tree changes and recent commits of one repo.
pub fn status(dir: &Path) -> Result<Value> {
    let branch = git_ok(dir, &["branch", "--show-current"])?.trim().to_string();
    let changes: Vec<Value> = git_ok(dir, &["status", "--porcelain", "-uall"])?
        .lines()
        .map(|l| json!({ "state": l.get(..2).unwrap_or("").trim(), "path": l.get(3..).unwrap_or("") }))
        .collect();
    let remotes: Vec<Value> = git_ok(dir, &["remote", "-v"])?
        .lines()
        .filter(|l| l.ends_with("(push)"))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some(json!({ "name": it.next()?, "url": it.next()? }))
        })
        .collect();
    let branches: Vec<String> = git_ok(dir, &["branch", "--format=%(refname:short)"])?
        .lines()
        .map(str::to_string)
        .collect();
    let commits: Vec<Value> = git_ok(dir, &["log", "-20", "--format=%h%x00%cI%x00%an%x00%s"])?
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\0').collect();
            (f.len() == 4).then(|| json!({ "sha": f[0], "date": f[1], "author": f[2], "subject": f[3] }))
        })
        .collect();
    Ok(json!({
        "path": dir, "branch": branch, "branches": branches, "remotes": remotes,
        "changes": changes, "commits": commits,
    }))
}

/// Subcommands that only read. Anything else must go through `exec` (confirmed).
const READ_ONLY: &[&str] = &[
    "status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files", "shortlog",
    "describe", "blame", "grep", "reflog", "ls-remote", "cat-file", "for-each-ref", "tag",
];
/// Flags that turn a read-only subcommand into a writing one.
const WRITING_FLAGS: &[&str] = &[
    "-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy", "--set-upstream-to",
    "add", "remove", "rm", "rename", "set-url", "prune", "-a", "-f", "--force",
];

pub fn read(dir: &Path, args: &[String]) -> Result<Value> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    // In log/show/diff, -c/-m/-M/-C are diff options, not writes.
    let history = matches!(sub, "log" | "show" | "diff");
    let writes = args.iter().skip(1).any(|a| {
        let l = a.to_lowercase();
        (!history && WRITING_FLAGS.contains(&a.as_str()))
            || l.starts_with("--output")
            || l.starts_with("--open-files-in-pager")
            || a.starts_with("-O")
            || RUNS_PROGRAMS.iter().any(|p| l.contains(p))
    });
    if !READ_ONLY.contains(&sub) || writes {
        return Err(TrackerError::BadArgs(format!(
            "`git {}` may change the repo; ask for it with git_exec, which a person confirms",
            args.join(" ")
        )));
    }
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let (ok, out, err) = git(dir, &argv)?;
    Ok(json!({ "ok": ok, "stdout": truncate(&out, 20_000), "stderr": truncate(&err, 4_000) }))
}

/// Options that make git run a program of the caller's choosing. A confirmed git
/// command should do git things, so these are refused outright.
const RUNS_PROGRAMS: &[&str] = &[
    "--upload-pack", "--receive-pack", "--exec", "core.sshcommand", "core.hookspath", "core.pager",
    "core.editor", "core.fsmonitor", "credential.helper", "protocol.ext.allow", "ext::",
];

pub fn exec(dir: &Path, args: &[String]) -> Result<Value> {
    let Some(first) = args.first() else {
        return Err(TrackerError::BadArgs("no git arguments".into()));
    };
    if first.starts_with('-') {
        return Err(TrackerError::BadArgs(format!(
            "git options before the subcommand ({first}) are not allowed; start with the subcommand"
        )));
    }
    if let Some(bad) = args
        .iter()
        .find(|a| RUNS_PROGRAMS.iter().any(|p| a.to_lowercase().contains(p)))
    {
        return Err(TrackerError::BadArgs(format!("`{bad}` would let git run another program; refused")));
    }
    if matches!(first.as_str(), "config" | "difftool" | "mergetool" | "filter-branch" | "bisect") {
        return Err(TrackerError::BadArgs(format!("`git {first}` is not available here; run it in a terminal")));
    }
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let (ok, out, err) = git(dir, &argv)?;
    Ok(json!({ "ok": ok, "command": format!("git {}", args.join(" ")), "stdout": truncate(&out, 20_000), "stderr": truncate(&err, 4_000) }))
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut cut = n;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}\n… ({} more bytes)", &s[..cut], s.len() - cut)
    }
}

/// Push `branch` to the remote whose host belongs to profile account `account`.
/// A repo with a `.sync.toml` must go through `sync_run` instead, which respects
/// what each origin may see.
pub fn push_branch(dir: &Path, branch: &str, account: &str) -> Result<Value> {
    if dir.join(crate::sync::manifest::FILE).exists() {
        return Err(TrackerError::BadArgs(
            "this repo has a .sync.toml: push with sync_run (remotes: [...]) so hidden paths stay hidden".into(),
        ));
    }
    let profile = Profile::require()?;
    let a = profile
        .account(account)
        .ok_or_else(|| TrackerError::Profile(format!("no account {account:?} in the profile")))?;
    let remotes = git_ok(dir, &["remote", "-v"])?;
    let remote = remotes
        .lines()
        .filter(|l| l.ends_with("(push)"))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .find(|(_, url)| host_of(url).as_deref() == Some(a.host.as_str()))
        .map(|(n, _)| n)
        .ok_or_else(|| {
            TrackerError::BadArgs(format!("this repo has no remote on {} (account {account})", a.host))
        })?;
    let (ok, out, err) = git(dir, &["push", &remote, branch])?;
    Ok(json!({ "ok": ok, "remote": remote, "branch": branch, "account": account, "stdout": out, "stderr": err }))
}

/// `owner/name` of the repo's GitHub remote.
fn github_repo(dir: &Path) -> Result<String> {
    let remotes = git_ok(dir, &["remote", "-v"])?;
    for l in remotes.lines() {
        let url = l.split_whitespace().nth(1).unwrap_or("");
        if host_of(url).as_deref() == Some("github.com") {
            let path = url
                .split_once("github.com")
                .map(|(_, p)| p.trim_start_matches([':', '/']))
                .unwrap_or("");
            return Ok(path.trim_end_matches(".git").trim_end_matches('/').to_string());
        }
    }
    Err(TrackerError::BadArgs("this repo has no GitHub remote, so it has no Codespace".into()))
}

fn github_token() -> Result<String> {
    let p = Profile::require()?;
    let a = p
        .accounts
        .iter()
        .find(|a| a.kind == Forge::Github)
        .ok_or_else(|| TrackerError::Profile("no GitHub account in the profile".into()))?;
    crate::profile::forge::credential(&a.host, &a.user)?
        .ok_or_else(|| TrackerError::Keeper("no GitHub credential from git (KeePassXC or its fallback)".into()))
}

fn gh(method: &str, path: &str, token: &str, body: Option<Value>) -> Result<(u16, Value)> {
    let url = format!("https://api.github.com{path}");
    let req = ureq::request(method, &url)
        .set("Authorization", &format!("Bearer {token}"))
        .set("Accept", "application/vnd.github+json")
        .set("User-Agent", "tracker");
    let resp = match body {
        Some(b) => req.send_json(b),
        None => req.call(),
    };
    match resp {
        Ok(r) => {
            let s = r.status();
            Ok((s, r.into_json().unwrap_or(Value::Null)))
        }
        Err(ureq::Error::Status(s, r)) => Ok((s, r.into_json().unwrap_or(Value::Null))),
        Err(e) => Err(TrackerError::Forge(format!("{url}: {e}"))),
    }
}

/// Open the repo's Codespace: reuse one if it exists (starting it if stopped),
/// otherwise create one on `branch`. Returns its web URL.
pub fn codespace(dir: &Path, branch: Option<&str>) -> Result<Value> {
    let full = github_repo(dir)?;
    let token = github_token()?;
    let scope_hint = "the GitHub token lacks the `codespace` scope; make a new one with `tracker tokens set github`";
    let (s, list) = gh("GET", &format!("/repos/{full}/codespaces"), &token, None)?;
    if s == 401 || s == 403 {
        return Err(TrackerError::Forge(scope_hint.into()));
    }
    let existing = list["codespaces"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|c| branch.map_or(true, |b| c["git_status"]["ref"].as_str() == Some(b)))
        .cloned();
    if let Some(c) = existing {
        let name = c["name"].as_str().unwrap_or("");
        let state = c["state"].as_str().unwrap_or("");
        if state != "Available" && state != "Starting" {
            gh("POST", &format!("/user/codespaces/{name}/start"), &token, None)?;
        }
        return Ok(json!({ "repo": full, "name": name, "state": "starting or running", "web_url": c["web_url"], "created": false }));
    }
    let mut body = json!({});
    if let Some(b) = branch {
        body["ref"] = json!(b);
    }
    let (s, c) = gh("POST", &format!("/repos/{full}/codespaces"), &token, Some(body))?;
    match s {
        200 | 201 | 202 => Ok(json!({ "repo": full, "name": c["name"], "state": c["state"], "web_url": c["web_url"], "created": true })),
        401 | 403 => Err(TrackerError::Forge(scope_hint.into())),
        _ => Err(TrackerError::Forge(format!("GitHub refused the Codespace ({s}): {}", c["message"]))),
    }
}

// ── browsing and editing a repo, as on a forge ───────────────────────────────

const MAX_TREE: usize = 20_000;
const MAX_TEXT: usize = 1 << 20;
const MAX_IMAGE: usize = 3 << 20;

/// A path inside the repo, or a refusal: no absolute paths, no `..`, nothing in `.git`.
fn inside(dir: &Path, rel: &str) -> Result<std::path::PathBuf> {
    let rel = rel.replace('\\', "/");
    let bad = rel.is_empty()
        || rel.starts_with('/')
        || rel.contains(':')
        || rel.split('/').any(|c| c == ".." || c == "." || c.eq_ignore_ascii_case(".git") || c.is_empty());
    if bad {
        return Err(TrackerError::BadArgs(format!("{rel:?} is not a path inside the repo")));
    }
    let full = dir.join(&rel);
    // Follow symlinks in the existing part of the path: it must stay inside.
    let root = dir.canonicalize()?;
    let mut probe = full.clone();
    while !probe.exists() {
        match probe.parent() {
            Some(p) => probe = p.to_path_buf(),
            None => break,
        }
    }
    if !probe.canonicalize()?.starts_with(&root) {
        return Err(TrackerError::BadArgs(format!("{rel:?} leads outside the repo")));
    }
    Ok(full)
}

/// Working-tree state per path from `git status --porcelain`.
fn changes(dir: &Path) -> Result<Vec<(String, String)>> {
    Ok(git_ok(dir, &["status", "--porcelain", "-uall"])?
        .lines()
        .filter(|l| l.len() > 3)
        .map(|l| {
            let path = l[3..].rsplit(" -> ").next().unwrap_or("").trim_matches('"').to_string();
            (path, l[..2].trim().to_string())
        })
        .collect())
}

/// Every file at `rev` (a branch, tag or commit), or in the working tree when none:
/// tracked plus untracked-but-not-ignored, each with its change state.
pub fn tree(dir: &Path, rev: Option<&str>) -> Result<Value> {
    let (files, state): (Vec<String>, Vec<(String, String)>) = match rev {
        Some(r) => (
            git_ok(dir, &["ls-tree", "-r", "--name-only", r])?.lines().map(str::to_string).collect(),
            Vec::new(),
        ),
        None => {
            let mut f: Vec<String> = git_ok(dir, &["ls-files", "--cached", "--others", "--exclude-standard"])?
                .lines()
                .map(str::to_string)
                .collect();
            f.sort();
            f.dedup();
            (f, changes(dir)?)
        }
    };
    let truncated = files.len() > MAX_TREE;
    let state: std::collections::HashMap<String, String> = state.into_iter().collect();
    let entries: Vec<Value> = files
        .into_iter()
        .take(MAX_TREE)
        .map(|p| match state.get(&p) {
            Some(s) => json!({ "path": p, "state": s }),
            None => json!({ "path": p }),
        })
        .collect();
    let deleted: Vec<&String> = state.iter().filter(|(_, s)| s.contains('D')).map(|(p, _)| p).collect();
    let branch = git_ok(dir, &["branch", "--show-current"])?.trim().to_string();
    Ok(json!({ "rev": rev, "branch": branch, "files": entries, "deleted": deleted, "truncated": truncated }))
}

fn image_type(path: &str) -> Option<&'static str> {
    let ext = Path::new(path).extension()?.to_str()?.to_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        _ => return None,
    })
}

/// One file, from the working tree or at `rev`. Text comes back as text; images as
/// base64 with their type; other binaries only by size.
pub fn file(dir: &Path, path: &str, rev: Option<&str>) -> Result<Value> {
    let bytes = match rev {
        Some(r) => {
            let out = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(["show", &format!("{r}:{}", path.replace('\\', "/"))])
                .output()
                .map_err(|e| TrackerError::GitMissing(e.to_string()))?;
            if !out.status.success() {
                return Err(TrackerError::BadArgs(format!("{path} does not exist at {r}")));
            }
            out.stdout
        }
        None => std::fs::read(inside(dir, path)?)
            .map_err(|e| TrackerError::BadArgs(format!("cannot read {path}: {e}")))?,
    };
    let size = bytes.len();
    if let Some(mime) = image_type(path) {
        if size <= MAX_IMAGE {
            use base64::Engine;
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            return Ok(json!({ "path": path, "rev": rev, "size": size, "kind": "image", "mime": mime, "base64": data }));
        }
    }
    let binary = bytes.iter().take(8000).any(|b| *b == 0);
    if binary || size > MAX_TEXT {
        return Ok(json!({ "path": path, "rev": rev, "size": size, "kind": if binary { "binary" } else { "too_large" } }));
    }
    Ok(json!({ "path": path, "rev": rev, "size": size, "kind": "text", "text": String::from_utf8_lossy(&bytes) }))
}

/// Uncommitted changes (staged and not), optionally of one path; or one commit's diff.
pub fn diff(dir: &Path, commit: Option<&str>, path: Option<&str>) -> Result<Value> {
    let mut args: Vec<String> = match commit {
        Some(c) => vec!["show".into(), "--format=%H%n%an <%ae>%n%cI%n%B%n--".into(), "--stat".into(), "--patch".into(), c.into()],
        None => vec!["diff".into(), "HEAD".into(), "--patch".into(), "--stat".into()],
    };
    if let Some(p) = path {
        args.push("--".into());
        args.push(p.into());
    }
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let (ok, out, err) = git(dir, &argv)?;
    if !ok && commit.is_none() {
        // A repo with no commit yet has no HEAD to diff against.
        let (_, out, _) = git(dir, &["diff", "--cached", "--patch"])?;
        return Ok(json!({ "commit": commit, "diff": truncate(&out, 400_000) }));
    }
    if !ok {
        return Err(TrackerError::BadArgs(err.trim().to_string()));
    }
    let mut untracked = Vec::new();
    if commit.is_none() {
        untracked = changes(dir)?.into_iter().filter(|(_, s)| s == "??").map(|(p, _)| p).collect();
    }
    Ok(json!({ "commit": commit, "diff": truncate(&out, 400_000), "untracked": untracked }))
}

/// Commits reachable from `rev` (default HEAD), optionally touching `path`.
pub fn log(dir: &Path, rev: Option<&str>, path: Option<&str>, limit: usize) -> Result<Value> {
    let n = format!("-{}", limit.clamp(1, 500));
    let mut args = vec!["log", n.as_str(), "--format=%H%x00%h%x00%cI%x00%an%x00%s%x00%D", rev.unwrap_or("HEAD")];
    if let Some(p) = path {
        args.push("--");
        args.push(p);
    }
    let (ok, out, _) = git(dir, &args)?;
    let commits: Vec<Value> = if ok {
        out.lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split('\0').collect();
                (f.len() == 6).then(|| json!({ "sha": f[0], "short": f[1], "date": f[2], "author": f[3], "subject": f[4], "refs": f[5] }))
            })
            .collect()
    } else {
        Vec::new()
    };
    Ok(json!({ "rev": rev, "path": path, "commits": commits }))
}

/// Write a file in the working tree (creating folders as needed), or delete it.
pub fn write(dir: &Path, path: &str, content: Option<&str>) -> Result<Value> {
    let full = inside(dir, path)?;
    match content {
        Some(text) => {
            if let Some(p) = full.parent() {
                std::fs::create_dir_all(p)?;
            }
            // Keep the file's line endings if it has CRLF ones.
            let crlf = std::fs::read(&full).map(|b| b.windows(2).any(|w| w == b"\r\n")).unwrap_or(false);
            let text = if crlf { text.replace("\r\n", "\n").replace('\n', "\r\n") } else { text.to_string() };
            std::fs::write(&full, text.as_bytes())?;
            Ok(json!({ "path": path, "written": text.len() }))
        }
        None => {
            std::fs::remove_file(&full).map_err(|e| TrackerError::BadArgs(format!("cannot delete {path}: {e}")))?;
            Ok(json!({ "path": path, "deleted": true }))
        }
    }
}

/// Stage `paths` (default: every change) and commit them, first switching to
/// `branch` — created from the current commit if it does not exist.
pub fn commit(dir: &Path, message: &str, paths: &[String], branch: Option<&str>) -> Result<Value> {
    if message.trim().is_empty() {
        return Err(TrackerError::BadArgs("a commit needs a message".into()));
    }
    if let Some(b) = branch {
        let current = git_ok(dir, &["branch", "--show-current"])?.trim().to_string();
        if current != b {
            let exists = git(dir, &["rev-parse", "--verify", "--quiet", &format!("refs/heads/{b}")])?.0;
            // Switching carries uncommitted changes along, which is what is wanted here.
            if exists {
                git_ok(dir, &["switch", b])?;
            } else {
                git_ok(dir, &["switch", "-c", b])?;
            }
        }
    }
    if paths.is_empty() {
        git_ok(dir, &["add", "--all"])?;
    } else {
        for p in paths {
            inside(dir, p)?;
        }
        let mut args = vec!["add", "--all", "--"];
        args.extend(paths.iter().map(String::as_str));
        git_ok(dir, &args)?;
    }
    let (ok, out, err) = git(dir, &["commit", "-m", message])?;
    if !ok {
        return Err(TrackerError::GitFailed { args: "commit".into(), code: 1, stderr: format!("{}{}", out.trim(), err.trim()) });
    }
    let sha = git_ok(dir, &["rev-parse", "HEAD"])?.trim().to_string();
    let branch = git_ok(dir, &["branch", "--show-current"])?.trim().to_string();
    Ok(json!({ "ok": true, "sha": sha, "branch": branch, "stdout": out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for a in [&["init", "-q", "-b", "main"][..], &["config", "user.email", "t@t"], &["config", "user.name", "t"]] {
            Command::new("git").args(a).current_dir(d.path()).status().unwrap();
        }
        d
    }

    #[test]
    fn paths_must_stay_inside_the_repo() {
        let d = repo();
        for bad in ["../x", "/etc/passwd", "a/../../b", ".git/config", "C:/x", "a//b", ""] {
            assert!(inside(d.path(), bad).is_err(), "{bad}");
        }
        assert!(inside(d.path(), "src/new/file.rs").is_ok());
    }

    #[test]
    fn write_then_commit_on_a_new_branch() {
        let d = repo();
        write(d.path(), "a.txt", Some("one\n")).unwrap();
        commit(d.path(), "first", &[], None).unwrap();
        write(d.path(), "dir/b.txt", Some("two\n")).unwrap();
        let t = tree(d.path(), None).unwrap();
        assert!(t["files"].as_array().unwrap().iter().any(|f| f["path"] == "dir/b.txt" && f["state"] == "??"));
        assert!(diff(d.path(), None, None).unwrap()["untracked"].as_array().unwrap().iter().any(|p| p == "dir/b.txt"));
        let c = commit(d.path(), "second", &["dir/b.txt".into()], Some("feature")).unwrap();
        assert_eq!(c["branch"], "feature");
        assert_eq!(file(d.path(), "dir/b.txt", Some("feature")).unwrap()["text"], "two\n");
        assert!(file(d.path(), "dir/b.txt", Some("main")).is_err());
        assert_eq!(log(d.path(), None, None, 10).unwrap()["commits"].as_array().unwrap().len(), 2);
        assert!(commit(d.path(), " ", &[], None).is_err());
    }
}
