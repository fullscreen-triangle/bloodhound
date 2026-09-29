//! The visibility manifest: which origins a repo has, and which paths each may see.
//!
//! It lives at `.sync.toml` on the canonical branch and is itself visible to no
//! remote, since it names the hidden paths. A path is visible to a remote iff the
//! remote `sees` every label whose glob matches the path; paths no glob matches are
//! visible everywhere. A label no remote sees never leaves the canonical repo.

use crate::error::{Result, TrackerError};
use globset::{GlobBuilder, GlobMatcher};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FILE: &str = ".sync.toml";

pub const TEMPLATE: &str = r#"# tracker sync — visibility manifest.
# Commit this on the canonical branch. It is never visible to any remote.

branch = "main"   # the canonical branch that is synced

# Each remote sees every unlabelled path, plus the labels it lists in `sees`.
[remotes.github]
url  = "git@github.com:you/project.git"
sees = []

[remotes.uni]
url  = "git@gitlab.uni-greifswald.de:you/project.git"
sees = ["uni"]

[remotes.bitspark]
url  = "git@gitlab.bitspark.de:you/project.git"
sees = ["uni", "bitspark"]
# branch = "main"             # remote branch, if it differs from the canonical one
# mixed_messages = "refuse"   # refuse | generic | keep  (see README)
# account = "bitspark"        # profile account whose identity it sees (default: by host)

# glob = label. Globs match the full path from the repo root; a trailing "/"
# means "everything under this directory". A path carrying several labels is
# visible only to remotes that see all of them.
[hide]
"bitspark/"      = "bitspark"
"notes/private/" = "private"
"#;

/// What to do when a commit changes both visible and hidden paths, so its message
/// may describe work the remote must not see.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MixedMessages {
    /// Stop and ask for an explicit message for this remote.
    #[default]
    Refuse,
    /// Replace the message with a neutral one.
    Generic,
    /// Keep the original message.
    Keep,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    #[serde(default = "default_branch")]
    branch: String,
    #[serde(default)]
    remotes: BTreeMap<String, RawRemote>,
    #[serde(default)]
    hide: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRemote {
    url: String,
    #[serde(default)]
    sees: Vec<String>,
    branch: Option<String>,
    #[serde(default)]
    mixed_messages: MixedMessages,
    account: Option<String>,
}

fn default_branch() -> String {
    "main".into()
}

#[derive(Debug, Clone, Serialize)]
pub struct Remote {
    pub name: String,
    pub url: String,
    pub branch: String,
    pub sees: BTreeSet<String>,
    pub mixed_messages: MixedMessages,
    /// Profile account whose identity this origin sees on your commits
    /// (default: the account whose host serves `url`).
    pub account: Option<String>,
}

#[derive(Debug)]
struct Rule {
    pattern: String,
    label: String,
    matcher: GlobMatcher,
}

#[derive(Debug)]
pub struct Manifest {
    pub branch: String,
    pub remotes: Vec<Remote>,
    rules: Vec<Rule>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Manifest> {
        let raw: RawManifest =
            toml::from_str(text).map_err(|e| TrackerError::Manifest(format!("{FILE}: {e}")))?;

        let mut remotes = Vec::new();
        for (name, r) in raw.remotes {
            let valid = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !valid {
                return Err(TrackerError::Manifest(format!(
                    "remote name {name:?} may only contain letters, digits, '-' and '_'"
                )));
            }
            remotes.push(Remote {
                branch: r.branch.unwrap_or_else(|| raw.branch.clone()),
                name,
                url: r.url,
                sees: r.sees.into_iter().collect(),
                mixed_messages: r.mixed_messages,
                account: r.account,
            });
        }

        let mut rules = Vec::new();
        for (pattern, label) in raw.hide {
            let mut glob = pattern.trim_start_matches('/').to_string();
            if glob.ends_with('/') {
                glob.push_str("**");
            }
            let matcher = GlobBuilder::new(&glob)
                .literal_separator(true)
                .build()
                .map_err(|e| TrackerError::Manifest(format!("hide pattern {pattern:?}: {e}")))?
                .compile_matcher();
            rules.push(Rule { pattern, label, matcher });
        }

        Ok(Manifest {
            branch: raw.branch,
            remotes,
            rules,
        })
    }

    pub fn visible(&self, remote: &Remote, path: &str) -> bool {
        path != FILE
            && self
                .rules
                .iter()
                .filter(|r| r.matcher.is_match(path))
                .all(|r| remote.sees.contains(&r.label))
    }

    /// `(glob, label)` for every hide rule, as written in the manifest.
    pub fn hide_rules(&self) -> Vec<(&str, &str)> {
        self.rules.iter().map(|r| (r.pattern.as_str(), r.label.as_str())).collect()
    }

    pub fn remote(&self, name: &str) -> Result<&Remote> {
        self.remotes
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| TrackerError::Manifest(format!("no remote named {name:?} in {FILE}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest::parse(
            r#"
            [remotes.pub]
            url = "a"
            [remotes.uni]
            url = "b"
            sees = ["uni"]
            [remotes.co]
            url = "c"
            sees = ["uni", "co"]
            [hide]
            "co/" = "co"
            "*.key" = "private"
            "shared/uni-only.md" = "uni"
            "co/joint/" = "uni"
            "#,
        )
        .unwrap()
    }

    #[test]
    fn unlabelled_paths_are_visible_everywhere() {
        let m = manifest();
        for r in &m.remotes {
            assert!(m.visible(r, "src/lib.rs"));
        }
    }

    #[test]
    fn labels_gate_by_sees() {
        let m = manifest();
        let (pubr, uni, co) = (m.remote("pub").unwrap(), m.remote("uni").unwrap(), m.remote("co").unwrap());
        assert!(!m.visible(pubr, "co/pricing.md"));
        assert!(!m.visible(uni, "co/pricing.md"));
        assert!(m.visible(co, "co/deep/pricing.md"));
        assert!(!m.visible(pubr, "shared/uni-only.md"));
        assert!(m.visible(uni, "shared/uni-only.md"));
    }

    #[test]
    fn a_path_with_several_labels_needs_all_of_them() {
        let m = manifest();
        // co/joint/x carries both "co" and "uni": uni lacks "co".
        assert!(!m.visible(m.remote("uni").unwrap(), "co/joint/x"));
        assert!(m.visible(m.remote("co").unwrap(), "co/joint/x"));
    }

    #[test]
    fn star_does_not_cross_directories_and_unseen_labels_never_leave() {
        let m = manifest();
        let co = m.remote("co").unwrap();
        assert!(!m.visible(co, "server.key"));
        assert!(m.visible(co, "certs/server.key"));
    }

    #[test]
    fn the_manifest_itself_is_never_visible() {
        let m = manifest();
        for r in &m.remotes {
            assert!(!m.visible(r, FILE));
        }
    }

    #[test]
    fn template_parses() {
        let m = Manifest::parse(TEMPLATE).unwrap();
        assert_eq!(m.remotes.len(), 3);
    }

    #[test]
    fn bad_remote_names_are_rejected() {
        assert!(Manifest::parse("[remotes.\"a b\"]\nurl = \"x\"\n").is_err());
    }
}
