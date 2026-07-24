//! The federation registry: persisted tracker state.
//!
//! The tracker OWNS this (χ, the committed count m, salient surface, the repo list).
//! It BORROWS the self-graph from `purpose` and execution from network-yield.
//!
//! Invariant I2 (never-resetting committed count, T4): `committed` only ever
//! increases. No operation here decrements it; `record_act` is the sole mutator and
//! it only adds. Loading + saving round-trips it faithfully.

use crate::error::{Result, TrackerError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Directory name holding the federation state, at the federation root.
pub const DIR: &str = ".tracker";
const FILE: &str = "federation.json";

/// A single tracked repo's persisted record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoRecord {
    pub name: String,
    /// Absolute path to the repo root on this machine.
    pub path: PathBuf,
    /// Optional remote (e.g. GitHub URL), for federation-level identity.
    #[serde(default)]
    pub remote: Option<String>,
    /// Last-computed χ (change-detector only; answers come fresh from search — I3).
    #[serde(default)]
    pub chi: Option<f64>,
    /// The committed count m (T4, I2). Monotone, never decremented.
    #[serde(default)]
    pub committed: u64,
}

impl RepoRecord {
    /// Record one committed tracked act (an index, a χ recompute, a run).
    /// The ONLY way `committed` changes — and it only goes up (I2).
    pub fn record_act(&mut self) {
        self.committed += 1;
    }
}

/// The whole federation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Federation {
    #[serde(default)]
    pub repos: Vec<RepoRecord>,
    /// Where this state lives (not serialised; set on load).
    #[serde(skip)]
    root: PathBuf,
}

impl Federation {
    /// State directory path given a federation root.
    fn dir_at(root: &Path) -> PathBuf {
        root.join(DIR)
    }
    fn file_at(root: &Path) -> PathBuf {
        Self::dir_at(root).join(FILE)
    }

    /// Create a fresh federation at `root`. Errors if one already exists.
    pub fn init(root: &Path) -> Result<Federation> {
        let file = Self::file_at(root);
        if file.exists() {
            return Err(TrackerError::AlreadyInitialised(root.to_path_buf()));
        }
        std::fs::create_dir_all(Self::dir_at(root))?;
        let fed = Federation {
            repos: Vec::new(),
            root: root.to_path_buf(),
        };
        fed.save()?;
        Ok(fed)
    }

    /// Discover and load the federation by walking up from `start` to find `.tracker/`.
    pub fn discover(start: &Path) -> Result<Federation> {
        let mut cur = Some(start);
        while let Some(dir) = cur {
            if Self::file_at(dir).exists() {
                return Self::load(dir);
            }
            cur = dir.parent();
        }
        Err(TrackerError::NoFederation(start.to_path_buf()))
    }

    fn load(root: &Path) -> Result<Federation> {
        let path = Self::file_at(root);
        let text = std::fs::read_to_string(&path)?;
        let mut fed: Federation =
            serde_json::from_str(&text).map_err(|source| TrackerError::StateCorrupt {
                path: path.clone(),
                source,
            })?;
        fed.root = root.to_path_buf();
        Ok(fed)
    }

    /// Persist state atomically (write temp, then rename).
    pub fn save(&self) -> Result<()> {
        let dir = Self::dir_at(&self.root);
        std::fs::create_dir_all(&dir)?;
        let final_path = Self::file_at(&self.root);
        let tmp = dir.join(format!("{FILE}.tmp"));
        let json = serde_json::to_string_pretty(self).expect("federation serialises");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &final_path)?;
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn get(&self, name: &str) -> Result<&RepoRecord> {
        self.repos
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| TrackerError::UnknownRepo(name.to_string()))
    }

    pub fn get_mut(&mut self, name: &str) -> Result<&mut RepoRecord> {
        self.repos
            .iter_mut()
            .find(|r| r.name == name)
            .ok_or_else(|| TrackerError::UnknownRepo(name.to_string()))
    }

    /// Register a new repo. Errors on duplicate name.
    pub fn add(&mut self, record: RepoRecord) -> Result<()> {
        if self.repos.iter().any(|r| r.name == record.name) {
            return Err(TrackerError::DuplicateRepo(record.name));
        }
        self.repos.push(record);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_count_only_increases() {
        let mut r = RepoRecord {
            name: "x".into(),
            path: PathBuf::from("."),
            remote: None,
            chi: None,
            committed: 0,
        };
        r.record_act();
        r.record_act();
        assert_eq!(r.committed, 2);
        // There is deliberately no API to lower it (I2).
    }

    #[test]
    fn init_discover_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mut fed = Federation::init(root).unwrap();
        fed.add(RepoRecord {
            name: "alpha".into(),
            path: root.join("alpha"),
            remote: None,
            chi: Some(3.0),
            committed: 5,
        })
        .unwrap();
        fed.save().unwrap();

        // discover from a nested subdir
        let nested = root.join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();
        let found = Federation::discover(&nested).unwrap();
        assert_eq!(found.repos.len(), 1);
        assert_eq!(found.get("alpha").unwrap().committed, 5);
    }

    #[test]
    fn double_init_errors() {
        let tmp = tempfile::tempdir().unwrap();
        Federation::init(tmp.path()).unwrap();
        assert!(Federation::init(tmp.path()).is_err());
    }
}
