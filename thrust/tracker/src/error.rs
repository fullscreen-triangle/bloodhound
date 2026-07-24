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

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("state file at {path} is corrupt: {source}")]
    StateCorrupt {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}
