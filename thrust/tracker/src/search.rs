//! Searching inside a repo's files, by composing `spraypaint` (0.2+).
//!
//! spraypaint ranks 40-line passages by the words in them and, more usefully here,
//! says whether the repo covers a query at all: `covered`, `partial` or `declined`
//! (see graffiti's `specifications.md`). The tracker adds three things around it:
//!
//! * the index is built on first use and rebuilt after new commits, and `.spraypaint/`
//!   goes into the repo's local `.git/info/exclude`, so the cache never shows up as a
//!   change to commit;
//! * searches are dry runs unless the caller asks to commit one, so exploring never
//!   raises the repo's committed count;
//! * results are shown to people and sometimes to hosted models, so passages from
//!   secret-bearing paths are dropped and key-shaped strings in the rest are masked.

use crate::error::{Result, TrackerError};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;
use std::time::UNIX_EPOCH;

fn spraypaint(dir: &Path, args: &[&str]) -> Result<Value> {
    let out = Command::new("spraypaint")
        .args(args)
        .arg("--root")
        .arg(dir)
        .output()
        .map_err(|e| TrackerError::Forge(format!("could not run spraypaint ({e}); `cargo install --path spraypaint` in graffiti")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(TrackerError::Forge(format!(
            "spraypaint failed: {}",
            err.lines().filter(|l| !l.trim().is_empty()).last().unwrap_or("").trim()
        )));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| TrackerError::Forge(format!("spraypaint output unreadable: {e}")))
}

fn mtime(p: &Path) -> Option<u64> {
    std::fs::metadata(p).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

fn last_commit(dir: &Path) -> Option<u64> {
    let out = Command::new("git").arg("-C").arg(dir).args(["log", "-1", "--format=%ct"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Keep `.spraypaint/` out of `git status` without touching any tracked file.
fn exclude_cache(dir: &Path) {
    let info = dir.join(".git").join("info");
    if !info.is_dir() && std::fs::create_dir_all(&info).is_err() {
        return; // `.git` is a file (worktree or submodule): leave it alone
    }
    let exclude = info.join("exclude");
    let current = std::fs::read_to_string(&exclude).unwrap_or_default();
    if !current.lines().any(|l| l.trim() == ".spraypaint/" || l.trim() == ".spraypaint") {
        let mut text = current;
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("# search cache written by tracker (spraypaint)\n.spraypaint/\n");
        let _ = std::fs::write(exclude, text);
    }
}

/// Build the index if there is none, or if the repo has commits newer than it.
pub fn ensure_index(dir: &Path, force: bool) -> Result<Option<Value>> {
    let index = dir.join(".spraypaint").join("index.json");
    let stale = match (mtime(&index), last_commit(dir)) {
        (None, _) => true,
        (Some(built), Some(commit)) => commit > built,
        (Some(_), None) => false,
    };
    if !(force || stale) {
        return Ok(None);
    }
    if !dir.join(".git").exists() {
        return Err(TrackerError::NotGitRepo(dir.to_path_buf()));
    }
    exclude_cache(dir);
    spraypaint(dir, &["index", "--json"]).map(Some)
}

/// Paths whose passages are never shown: they tend to hold keys and tokens.
fn sensitive(path: &str) -> bool {
    let p = path.to_lowercase();
    let name = p.rsplit('/').next().unwrap_or(&p);
    p.split('/').any(|c| c == ".claude" || c == ".ssh" || c == ".aws" || c == ".gnupg" || c == "secrets")
        || name == ".env"
        || name.starts_with(".env.")
        || name.ends_with(".local.json")
        || name.ends_with(".pem")
        || name.ends_with(".key")
        || name.starts_with("id_rsa")
        || name.starts_with("id_ed25519")
        || matches!(name, ".npmrc" | ".pypirc" | ".netrc" | "credentials" | "credentials.json" | "serve.token")
}

/// Mask anything shaped like an API key or token.
fn redact(text: &str) -> String {
    const PREFIXES: &[&str] = &[
        "sk-ant-", "sk-proj-", "hf_", "ghp_", "gho_", "ghs_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "AKIA", "AIza",
    ];
    text.lines()
        .map(|line| {
            if line.contains("-----BEGIN") && line.contains("PRIVATE KEY") {
                return "[private key redacted]".to_string();
            }
            let mut out = String::new();
            let mut rest = line;
            'scan: while !rest.is_empty() {
                for pre in PREFIXES {
                    if let Some(tail) = rest.strip_prefix(pre) {
                        let n = tail.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-')).unwrap_or(tail.len());
                        // Short tails are ordinary words ("hf_x" in prose), not keys.
                        if n >= 12 {
                            out.push_str(pre);
                            out.push_str("…[redacted]");
                            rest = &tail[n..];
                            continue 'scan;
                        }
                    }
                }
                let ch = rest.chars().next().expect("not empty");
                out.push(ch);
                rest = &rest[ch.len_utf8()..];
            }
            out
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub struct SearchOptions<'a> {
    pub query: &'a str,
    pub k: usize,
    pub scenes: &'a [String],
    pub commit: bool,
    pub refresh: bool,
}

/// Search one repo. Returns spraypaint's answer with secret-bearing passages
/// removed, plus what the tracker did around it.
pub fn search(dir: &Path, o: &SearchOptions) -> Result<Value> {
    if o.query.trim().is_empty() {
        return Err(TrackerError::BadArgs("an empty query searches nothing".into()));
    }
    let indexed = ensure_index(dir, o.refresh)?;
    let k = o.k.clamp(1, 50).to_string();
    let scenes = o.scenes.join(",");
    let mut args = vec!["ask", o.query, "--json", "-k", k.as_str()];
    if !o.commit {
        args.push("--dry-run");
    }
    if !scenes.is_empty() {
        args.push("--scenes");
        args.push(scenes.as_str());
    }
    let mut v = spraypaint(dir, &args)?;
    if v.get("coverage").is_none() {
        return Err(TrackerError::Forge(
            "this spraypaint build has no coverage verdict; install 0.2.0 or later from graffiti".into(),
        ));
    }
    let mut withheld = 0;
    if let Some(results) = v["results"].as_array_mut() {
        results.retain(|r| {
            let keep = !sensitive(r["path"].as_str().unwrap_or(""));
            withheld += usize::from(!keep);
            keep
        });
        for r in results.iter_mut() {
            if let Some(s) = r["snippet"].as_str() {
                r["snippet"] = json!(redact(s));
            }
        }
    }
    v["withheld"] = json!(withheld);
    v["reindexed"] = json!(indexed.is_some());
    v["repo_path"] = json!(dir);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_bearing_paths_are_withheld() {
        for p in [".env", "web/.env.local", ".claude/settings.local.json", "config/app.local.json", "keys/server.pem", "secrets/db.toml"] {
            assert!(sensitive(p), "{p}");
        }
        for p in ["src/env.rs", "README.md", "docs/secret-sharing.md", "tokens.rs"] {
            assert!(!sensitive(p), "{p}");
        }
    }

    #[test]
    fn key_shaped_strings_are_masked() {
        let t = "key = \"sk-ant-api03-abcdefghijklmnop\"\nsee hf_x and ghp_ABCDEFGHIJKLMNOPQRST!";
        let r = redact(t);
        assert!(!r.contains("abcdefghijklmnop") && !r.contains("ABCDEFGHIJKLMNOPQRST"), "{r}");
        assert!(r.contains("hf_x"), "short words stay: {r}");
        assert_eq!(redact("-----BEGIN OPENSSH PRIVATE KEY-----"), "[private key redacted]");
    }

    #[test]
    fn a_search_builds_the_index_and_keeps_the_cache_out_of_git() {
        if Command::new("spraypaint").arg("--version").output().is_err() {
            return; // not installed here
        }
        let d = tempfile::tempdir().unwrap();
        Command::new("git").args(["init", "-q"]).current_dir(d.path()).status().unwrap();
        std::fs::create_dir_all(d.path().join("docs")).unwrap();
        std::fs::write(d.path().join("docs/a.md"), "# Allocation\n\nwater filling clears a price\n").unwrap();
        std::fs::write(d.path().join("b.md"), "unrelated words about gardens\n").unwrap();
        std::fs::write(d.path().join("app.local.json"), "{\"note\": \"water filling price\", \"key\": \"sk-ant-api03-abcdefghijklmnop\"}\n").unwrap();
        let o = |q| SearchOptions { query: q, k: 5, scenes: &[], commit: false, refresh: false };
        let v = search(d.path(), &o("water filling price")).unwrap();
        assert_eq!(v["coverage"]["verdict"], "covered");
        assert!(v["results"].as_array().unwrap().iter().all(|r| r["path"] != "app.local.json"));
        assert_eq!(v["withheld"], 1);
        assert_eq!(v["dry_run"], true);
        let off = search(d.path(), &o("kubernetes helm")).unwrap();
        assert_eq!(off["coverage"]["verdict"], "declined");
        let status = Command::new("git").args(["status", "--porcelain"]).current_dir(d.path()).output().unwrap();
        assert!(!String::from_utf8_lossy(&status.stdout).contains(".spraypaint"));
    }
}
