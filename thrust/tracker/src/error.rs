//! Error types for the tracker.

use std::path::PathBuf;

/// The tracker's result alias.
pub type Result<T> = std::result::Result<T, TrackerError>;

#[derive(Debug, thiserror::Error)]
pub enum TrackerError {
    #[error("federation not initialised here; run `tracker init` (looked from {0})")]
    NoFederation(PathBuf),

    #[error("federation already initialised at {0}")]
    AlreadyInitialised(PathBuf),

    #[error("no repo named {0:?} in the federation")]
    UnknownRepo(String),

    #[error("repo {0:?} already registered")]
    DuplicateRepo(String),

    #[error("path does not exist or is not a directory: {0}")]
    BadRepoPath(PathBuf),

    #[error("the `purpose` search organ is not installed (expected on PATH). Install it, then retry. Underlying: {0}")]
    PurposeMissing(String),

    #[error("`purpose {args}` failed (exit {code}): {stderr}")]
    PurposeFailed {
        args: String,
        code: i32,
        stderr: String,
    },

    #[error("could not read `.purpose/index.json` for repo {repo:?}: {source}. Run `purpose index` in that repo (or `tracker add` re-indexes).")]
    IndexUnreadable {
        repo: String,
        #[source]
        source: std::io::Error,
    },

    #[error("`.purpose/index.json` for repo {repo:?} was not in the expected line format at line {line}: {detail}")]
    IndexMalformed {
        repo: String,
        line: usize,
        detail: String,
    },

    #[error("the network-yield execution organ is not installed; tracking/search work, but `run` is unavailable")]
    ExecutionMissing,

    #[error("could not run git ({0}); is it installed and on PATH?")]
    GitMissing(String),

    #[error("`git {args}` failed (exit {code}): {stderr}")]
    GitFailed {
        args: String,
        code: i32,
        stderr: String,
    },

    #[error("not inside a git repository: {0}")]
    NotGitRepo(PathBuf),

    #[error("sync manifest: {0}")]
    Manifest(String),

    #[error("{remote}: remote commit {commit} has no parent, so it is history this repo never produced. Adopting an existing, independent history is not supported yet.")]
    UnrelatedHistory { remote: String, commit: String },

    #[error("{remote}: remote commit {commit} changes {path}, which the manifest hides from {remote}. Run `tracker sync resolve {remote}` to apply it by hand, or change the manifest.")]
    HiddenFromRemote {
        remote: String,
        commit: String,
        path: String,
    },

    #[error("{remote}: remote commit {commit} conflicts with the canonical branch in: {paths}. Run `tracker sync resolve {remote}`, fix the conflicts, then `tracker sync resolve --continue`.")]
    SyncConflict {
        remote: String,
        commit: String,
        paths: String,
    },

    #[error("{remote}: nothing pushed — {count} commit(s) change both visible paths and paths hidden from {remote}, so their messages may describe hidden work:{commits}\nGive each the message {remote} should see with `tracker sync message <commit> --remote {remote} \"…\"` (or --shared), set `mixed_messages = \"generic\"` or `\"keep\"` for {remote} in .sync.toml, or rerun with --trust-messages.")]
    MessageLeak {
        remote: String,
        count: usize,
        commits: String,
    },

    #[error("{remote}: the remote has commits the canonical branch has not taken in; sync without --no-pull")]
    RemoteDiverged { remote: String },

    #[error("{remote}: leak check failed — a projected tree contains the hidden path {path}. Nothing was pushed. This is a bug in tracker.")]
    LeakCheck { remote: String, path: String },

    #[error("the canonical branch is checked out with uncommitted changes; commit or stash them first")]
    DirtyWorktree,

    #[error("a `tracker sync resolve` is in progress; finish it with `tracker sync resolve --continue` or `--abort`")]
    ResolveInProgress,

    #[error("{0}")]
    Resolve(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("profile: {0}")]
    Profile(String),

    #[error("forge: {0}")]
    Forge(String),

    #[error("KeePassXC: {0}")]
    Keeper(String),

    #[error("KeePassXC is locked; unlock it and try again")]
    KeeperLocked,

    #[error("bad arguments: {0}")]
    BadArgs(String),

    #[error("no operation named {0:?}; `tracker describe` lists them")]
    UnknownOperation(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("state file at {path} is corrupt: {source}")]
    StateCorrupt {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

impl TrackerError {
    /// A stable, machine-matchable name for the failure, for `--json` and MCP callers.
    pub fn code(&self) -> &'static str {
        use TrackerError::*;
        match self {
            NoFederation(_) => "no_federation",
            AlreadyInitialised(_) => "already_initialised",
            UnknownRepo(_) => "unknown_repo",
            DuplicateRepo(_) => "duplicate_repo",
            BadRepoPath(_) => "bad_repo_path",
            PurposeMissing(_) => "purpose_missing",
            PurposeFailed { .. } => "purpose_failed",
            IndexUnreadable { .. } => "index_unreadable",
            IndexMalformed { .. } => "index_malformed",
            ExecutionMissing => "execution_missing",
            GitMissing(_) => "git_missing",
            GitFailed { .. } => "git_failed",
            NotGitRepo(_) => "not_a_git_repo",
            Manifest(_) => "manifest",
            UnrelatedHistory { .. } => "unrelated_history",
            HiddenFromRemote { .. } => "hidden_from_remote",
            SyncConflict { .. } => "sync_conflict",
            MessageLeak { .. } => "message_leak",
            RemoteDiverged { .. } => "remote_diverged",
            LeakCheck { .. } => "leak_check",
            DirtyWorktree => "dirty_worktree",
            ResolveInProgress => "resolve_in_progress",
            Resolve(_) => "resolve",
            Internal(_) => "internal",
            Profile(_) => "profile",
            Forge(_) => "forge",
            BadArgs(_) => "bad_args",
            Keeper(_) => "keeper",
            KeeperLocked => "keeper_locked",
            UnknownOperation(_) => "unknown_operation",
            Io(_) => "io",
            StateCorrupt { .. } => "state_corrupt",
        }
    }
}
