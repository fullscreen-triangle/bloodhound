//! Thin wrapper over the `git` binary, plumbing only.
//!
//! Every method is one git subcommand, so object storage, merging and transport
//! behave exactly as git's own; nothing here reimplements them.

use crate::error::{Result, TrackerError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// An author/committer identity exactly as stored in the commit header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident {
    pub name: String,
    pub email: String,
    /// Raw `<unix-seconds> <tz>` form, which git accepts back verbatim.
    pub date: String,
}

impl Ident {
    fn parse(raw: &str) -> Option<Ident> {
        let lt = raw.find('<')?;
        let gt = lt + raw[lt..].find('>')?;
        Some(Ident {
            name: raw[..lt].trim_end().to_string(),
            email: raw[lt + 1..gt].to_string(),
            date: raw[gt + 1..].trim().to_string(),
        })
    }

    /// Key under which two commits count as "made by the same act".
    pub fn key(&self) -> String {
        format!("{}\u{0}{}\u{0}{}", self.name, self.email, self.date)
    }
}

pub fn act_key(author: &Ident, committer: &Ident) -> String {
    format!("{}\u{1}{}", author.key(), committer.key())
}

#[derive(Debug, Clone)]
pub struct Commit {
    pub sha: String,
    pub tree: String,
    pub parents: Vec<String>,
    pub author: Ident,
    pub committer: Ident,
    pub message: String,
}

impl Commit {
    pub fn subject(&self) -> &str {
        self.message.lines().next().unwrap_or("")
    }

    /// Identity of the act that made this commit: projection copies both idents
    /// verbatim, so a commit and its projection share this key.
    pub fn act_key(&self) -> String {
        act_key(&self.author, &self.committer)
    }
}

/// One `ls-tree -r` record, kept whole so it can be fed straight back to
/// `update-index --index-info`.
pub struct Entry {
    pub record: String,
    pub path: String,
}

pub struct Git {
    root: PathBuf,
    git_dir: PathBuf,
}

impl Git {
    pub fn open(path: &Path) -> Result<Git> {
        let probe = Git {
            root: path.to_path_buf(),
            git_dir: PathBuf::new(),
        };
        let out = probe.raw(&["rev-parse", "--show-toplevel", "--git-common-dir"], &[], None)?;
        if out.code != 0 {
            return Err(TrackerError::NotGitRepo(path.to_path_buf()));
        }
        let mut lines = out.stdout.lines();
        let root = PathBuf::from(lines.next().unwrap_or_default());
        let common = PathBuf::from(lines.next().unwrap_or_default());
        let git_dir = if common.is_absolute() {
            common
        } else {
            path.join(common)
        };
        Ok(Git { root, git_dir })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Private state for `tracker sync` inside the git dir (never part of any tree).
    pub fn sync_dir(&self) -> PathBuf {
        self.git_dir.join("tracker-sync")
    }

    pub fn raw(&self, args: &[&str], env: &[(&str, &str)], stdin: Option<&[u8]>) -> Result<Output> {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(&self.root).args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| TrackerError::GitMissing(e.to_string()))?;
        // Feed stdin from a thread so a chatty child can never deadlock on a full pipe.
        let writer = stdin.map(|data| {
            let mut pipe = child.stdin.take().expect("stdin was piped");
            let data = data.to_vec();
            std::thread::spawn(move || {
                let _ = pipe.write_all(&data);
            })
        });
        let out = child.wait_with_output()?;
        if let Some(w) = writer {
            let _ = w.join();
        }
        Ok(Output {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }

    pub fn run_with(&self, args: &[&str], env: &[(&str, &str)], stdin: Option<&[u8]>) -> Result<String> {
        let out = self.raw(args, env, stdin)?;
        if out.code != 0 {
            return Err(TrackerError::GitFailed {
                args: args.join(" "),
                code: out.code,
                stderr: out.stderr,
            });
        }
        Ok(out.stdout)
    }

    pub fn run(&self, args: &[&str]) -> Result<String> {
        self.run_with(args, &[], None)
    }

    /// Resolve `rev` to a commit sha, or `None` if it does not exist.
    pub fn rev(&self, rev: &str) -> Result<Option<String>> {
        let spec = format!("{rev}^{{commit}}");
        let out = self.raw(&["rev-parse", "--verify", "--quiet", &spec], &[], None)?;
        Ok((out.code == 0).then(|| out.stdout.trim().to_string()))
    }

    pub fn commit(&self, sha: &str) -> Result<Commit> {
        let raw = self.run(&["cat-file", "commit", sha])?;
        let (head, message) = raw.split_once("\n\n").unwrap_or((raw.as_str(), ""));
        let mut tree = None;
        let mut parents = Vec::new();
        let mut author = None;
        let mut committer = None;
        for line in head.lines() {
            // Continuation lines of multi-line headers (e.g. gpgsig) start with a space.
            if line.starts_with(' ') {
                continue;
            }
            match line.split_once(' ') {
                Some(("tree", v)) => tree = Some(v.to_string()),
                Some(("parent", v)) => parents.push(v.to_string()),
                Some(("author", v)) => author = Ident::parse(v),
                Some(("committer", v)) => committer = Ident::parse(v),
                _ => {}
            }
        }
        let bad = || TrackerError::GitFailed {
            args: format!("cat-file commit {sha}"),
            code: 0,
            stderr: "unparseable commit header".into(),
        };
        Ok(Commit {
            sha: sha.to_string(),
            tree: tree.ok_or_else(bad)?,
            parents,
            author: author.ok_or_else(bad)?,
            committer: committer.ok_or_else(bad)?,
            message: message.to_string(),
        })
    }

    pub fn ls_tree(&self, tree: &str) -> Result<Vec<Entry>> {
        let out = self.run(&["ls-tree", "-r", "-z", "--full-tree", tree])?;
        Ok(out
            .split('\0')
            .filter(|r| !r.is_empty())
            .map(|r| Entry {
                path: r.split_once('\t').map(|(_, p)| p).unwrap_or("").to_string(),
                record: r.to_string(),
            })
            .collect())
    }

    /// Build a tree object from `entries` using a scratch index, so the real index
    /// and working tree are never touched.
    pub fn write_tree(&self, entries: &[Entry]) -> Result<String> {
        let dir = self.sync_dir();
        std::fs::create_dir_all(&dir)?;
        let index = dir.join("scratch.index");
        let _ = std::fs::remove_file(&index);
        let index_str = index.to_string_lossy().into_owned();
        let env = [("GIT_INDEX_FILE", index_str.as_str())];
        let mut input = Vec::new();
        for e in entries {
            input.extend_from_slice(e.record.as_bytes());
            input.push(0);
        }
        self.run_with(&["update-index", "-z", "--index-info"], &env, Some(&input))?;
        let tree = self.run_with(&["write-tree"], &env, None)?;
        let _ = std::fs::remove_file(&index);
        Ok(tree.trim().to_string())
    }

    /// Create a commit object with the given identities copied verbatim.
    /// Unsigned on purpose: a signature would make projection non-deterministic.
    pub fn commit_tree(
        &self,
        tree: &str,
        parents: &[String],
        author: &Ident,
        committer: &Ident,
        message: &str,
    ) -> Result<String> {
        let mut args = vec!["commit-tree", "--no-gpg-sign", tree];
        for p in parents {
            args.push("-p");
            args.push(p);
        }
        args.extend(["-F", "-"]);
        let env = [
            ("GIT_AUTHOR_NAME", author.name.as_str()),
            ("GIT_AUTHOR_EMAIL", author.email.as_str()),
            ("GIT_AUTHOR_DATE", author.date.as_str()),
            ("GIT_COMMITTER_NAME", committer.name.as_str()),
            ("GIT_COMMITTER_EMAIL", committer.email.as_str()),
            ("GIT_COMMITTER_DATE", committer.date.as_str()),
        ];
        Ok(self
            .run_with(&args, &env, Some(message.as_bytes()))?
            .trim()
            .to_string())
    }

    /// Three-way merge of trees without touching the working tree.
    /// `Ok(tree)` when clean, `Err(conflicted paths)` otherwise.
    pub fn merge_tree(
        &self,
        base: &str,
        ours: &str,
        theirs: &str,
    ) -> Result<std::result::Result<String, Vec<String>>> {
        let base_arg = format!("--merge-base={base}");
        let args = [
            "merge-tree",
            "--write-tree",
            "--name-only",
            "--no-messages",
            base_arg.as_str(),
            ours,
            theirs,
        ];
        let out = self.raw(&args, &[], None)?;
        let mut lines = out.stdout.lines();
        let tree = lines.next().unwrap_or("").trim().to_string();
        match out.code {
            0 => Ok(Ok(tree)),
            1 => {
                let mut paths: Vec<String> = lines
                    .take_while(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect();
                paths.dedup();
                Ok(Err(paths))
            }
            code => Err(TrackerError::GitFailed {
                args: args.join(" "),
                code,
                stderr: out.stderr,
            }),
        }
    }

    /// Paths a commit changes relative to `parent` (or everything, for a root commit).
    pub fn changed_paths(&self, parent: Option<&str>, sha: &str) -> Result<Vec<String>> {
        let out = match parent {
            Some(p) => self.run(&["diff-tree", "-r", "-z", "--name-only", "--no-commit-id", p, sha])?,
            None => self.run(&["diff-tree", "-r", "-z", "--name-only", "--no-commit-id", "--root", sha])?,
        };
        Ok(out
            .split('\0')
            .filter(|p| !p.is_empty())
            .map(str::to_string)
            .collect())
    }

    /// All commits reachable from `tip`, parents before children.
    pub fn rev_list_topo(&self, tip: &str) -> Result<Vec<String>> {
        Ok(self
            .run(&["rev-list", "--reverse", "--topo-order", tip])?
            .lines()
            .map(str::to_string)
            .collect())
    }

    /// First-parent commits reachable from `tip` but not from any of `known`, oldest first.
    pub fn rev_list_new<'a>(&self, tip: &str, known: impl Iterator<Item = &'a String>) -> Result<Vec<String>> {
        let mut input = format!("{tip}\n");
        for k in known {
            input.push('^');
            input.push_str(k);
            input.push('\n');
        }
        Ok(self
            .run_with(
                &["rev-list", "--reverse", "--first-parent", "--ignore-missing", "--stdin"],
                &[],
                Some(input.as_bytes()),
            )?
            .lines()
            .map(str::to_string)
            .collect())
    }

    /// `(sha, tree, author, committer)` for every commit reachable from `tip`,
    /// parents before children, in one call.
    pub fn idents(&self, tip: &str) -> Result<Vec<(String, String, Ident, Ident)>> {
        let out = self.run(&[
            "log",
            "--topo-order",
            "--reverse",
            "--date=raw",
            "--format=%H%x00%T%x00%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd%x1e",
            tip,
        ])?;
        Ok(out
            .split('\u{1e}')
            .filter_map(|rec| {
                let f: Vec<&str> = rec.trim_start().split('\0').collect();
                (f.len() == 8).then(|| {
                    let ident = |n: &str, e: &str, d: &str| Ident {
                        name: n.to_string(),
                        email: e.to_string(),
                        date: d.to_string(),
                    };
                    (
                        f[0].to_string(),
                        f[1].to_string(),
                        ident(f[2], f[3], f[4]),
                        ident(f[5], f[6], f[7]),
                    )
                })
            })
            .collect())
    }

    /// `(sha, full message)` of every commit on `tip` whose message has a line
    /// starting with `prefix`.
    pub fn log_grep(&self, tip: &str, prefix: &str) -> Result<Vec<(String, String)>> {
        let grep = format!("--grep=^{prefix}");
        let out = self.run(&["log", &grep, "--format=%H%x00%B%x1e", tip])?;
        Ok(out
            .split('\u{1e}')
            .filter_map(|rec| {
                let (sha, body) = rec.trim_start().split_once('\0')?;
                Some((sha.to_string(), body.to_string()))
            })
            .collect())
    }

    pub fn is_ancestor(&self, ancestor: &str, descendant: &str) -> Result<bool> {
        let args = ["merge-base", "--is-ancestor", ancestor, descendant];
        let out = self.raw(&args, &[], None)?;
        match out.code {
            0 => Ok(true),
            1 => Ok(false),
            code => Err(TrackerError::GitFailed {
                args: args.join(" "),
                code,
                stderr: out.stderr,
            }),
        }
    }

    /// Current tip of `branch` on the remote at `url`, or `None` if it has no such branch.
    pub fn remote_head(&self, url: &str, branch: &str) -> Result<Option<String>> {
        let full = format!("refs/heads/{branch}");
        let out = self.run(&["ls-remote", url, &full])?;
        Ok(out.lines().find_map(|l| {
            let (sha, name) = l.split_once('\t')?;
            (name == full).then(|| sha.to_string())
        }))
    }

    pub fn fetch(&self, url: &str, branch: &str, into: &str) -> Result<()> {
        let spec = format!("+refs/heads/{branch}:{into}");
        self.run(&["fetch", "--no-tags", "--quiet", url, &spec])?;
        Ok(())
    }

    pub fn push(&self, url: &str, sha: &str, branch: &str) -> Result<()> {
        let spec = format!("{sha}:refs/heads/{branch}");
        self.run_with(
            &["push", "--quiet", url, &spec],
            &[(super::guard::PUSH_ENV, "1")],
            None,
        )?;
        Ok(())
    }

    pub fn update_ref(&self, name: &str, new: &str, old: Option<&str>) -> Result<()> {
        let mut args = vec!["update-ref", "-m", "tracker sync", name, new];
        if let Some(o) = old {
            args.push(o);
        }
        self.run(&args)?;
        Ok(())
    }

    /// The full ref HEAD points at, if HEAD is on a branch.
    pub fn head_branch(&self) -> Result<Option<String>> {
        let out = self.raw(&["symbolic-ref", "-q", "HEAD"], &[], None)?;
        Ok((out.code == 0).then(|| out.stdout.trim().to_string()))
    }

    /// Uncommitted changes to tracked files.
    pub fn is_dirty(&self) -> Result<bool> {
        Ok(!self
            .run(&["status", "--porcelain", "--untracked-files=no"])?
            .trim()
            .is_empty())
    }

    pub fn show_file(&self, rev: &str, path: &str) -> Result<Option<String>> {
        let spec = format!("{rev}:{path}");
        let out = self.raw(&["show", &spec], &[], None)?;
        Ok((out.code == 0).then_some(out.stdout))
    }

    /// Absolute path of `name` inside the git dir, honouring core.hooksPath etc.
    pub fn git_path(&self, name: &str) -> Result<PathBuf> {
        let p = PathBuf::from(self.run(&["rev-parse", "--git-path", name])?.trim());
        Ok(if p.is_absolute() { p } else { self.root.join(p) })
    }

    /// Commits annotated in notes ref `notes_ref`.
    pub fn noted(&self, notes_ref: &str) -> Result<std::collections::HashSet<String>> {
        let r = format!("--ref={notes_ref}");
        let out = self.raw(&["notes", &r, "list"], &[], None)?;
        if out.code != 0 {
            return Ok(Default::default());
        }
        Ok(out
            .stdout
            .lines()
            .filter_map(|l| l.split_whitespace().nth(1).map(str::to_string))
            .collect())
    }

    pub fn note(&self, notes_ref: &str, sha: &str) -> Result<Option<String>> {
        let r = format!("--ref={notes_ref}");
        let out = self.raw(&["notes", &r, "show", sha], &[], None)?;
        Ok((out.code == 0).then_some(out.stdout))
    }

    pub fn set_note(&self, notes_ref: &str, sha: &str, content: &str) -> Result<()> {
        let r = format!("--ref={notes_ref}");
        self.run_with(&["notes", &r, "add", "-f", "-F", "-", sha], &[], Some(content.as_bytes()))?;
        Ok(())
    }
}
