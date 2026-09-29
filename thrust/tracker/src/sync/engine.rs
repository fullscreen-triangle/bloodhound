//! The sync engine.
//!
//! **Pull.** Every remote commit the canonical branch has not seen is replayed onto
//! it: a three-way merge against the commit's own parent, so paths hidden from that
//! remote are carried through untouched. The replay keeps the collaborator's
//! identities and is tagged `Sync-Origin: <remote> <sha>`.
//!
//! **Push.** The canonical branch is projected onto each remote: every commit's tree
//! filtered by the manifest, identities and dates copied, unsigned. The same
//! canonical commit with the same projected parents always yields the same remote
//! commit, so a remote's history only ever grows and no push is ever forced.
//! A commit whose projection changes nothing is dropped. A replayed commit maps
//! straight back onto the remote commit it came from, or, if the canonical branch
//! moved meanwhile, becomes a merge with it.
//!
//! The canonical→remote map is cached in `.git/tracker-sync/<remote>.map`. It can
//! always be rebuilt: a projection copies both identities of its source commit, so
//! an unmapped remote commit whose identities match a canonical commit is adopted as
//! that commit's projection instead of being replayed.

use super::git::{act_key, Commit, Git, Ident};
use super::guard::{self, HookStatus};
use super::manifest::{self, Manifest, MixedMessages, Remote};
use crate::error::{Result, TrackerError};
use crate::profile::Profile;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

pub const ORIGIN: &str = "Sync-Origin";
pub const SHARED_MESSAGE: &str = "Shared-Message";
pub const NOTES_REF: &str = "sync-messages";
pub const RESOLVE_STATE: &str = "resolve";
const GENERIC_MESSAGE: &str = "Update shared files\n";

pub fn message_key(remote: &str) -> String {
    format!("Message-{remote}")
}

pub fn fetched_ref(remote: &str) -> String {
    format!("refs/sync/{remote}/remote")
}

fn projected_ref(remote: &str) -> String {
    format!("refs/sync/{remote}/projected")
}

pub fn short(sha: &str) -> &str {
    &sha[..sha.len().min(10)]
}

// ── messages ─────────────────────────────────────────────────────────────────

fn trailer_key(line: &str) -> Option<&str> {
    let (k, _) = line.split_once(": ")?;
    (!k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')).then_some(k)
}

fn is_sync_line(line: &str) -> bool {
    matches!(trailer_key(line), Some(k) if k == ORIGIN || k == SHARED_MESSAGE || k.starts_with("Message-"))
}

pub fn trailer_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(": ").map(str::trim))
}

/// The message with every sync-control line removed; these must never reach a remote.
pub fn clean_message(msg: &str) -> String {
    let kept: Vec<&str> = msg.lines().filter(|l| !is_sync_line(l)).collect();
    let body = kept.join("\n");
    let body = body.trim_end();
    if body.is_empty() {
        "(no message)\n".into()
    } else {
        format!("{body}\n")
    }
}

pub fn with_origin(msg: &str, remote: &str, sha: &str) -> String {
    let cleaned = clean_message(msg);
    let body = cleaned.trim_end();
    let paragraphs: Vec<&str> = body.split("\n\n").collect();
    let in_trailer_block = paragraphs.len() > 1
        && paragraphs
            .last()
            .is_some_and(|p| p.lines().all(|l| trailer_key(l).is_some()));
    let sep = if in_trailer_block { "\n" } else { "\n\n" };
    format!("{body}{sep}{ORIGIN}: {remote} {sha}\n")
}

// ── persisted canonical→remote map ───────────────────────────────────────────

struct Map {
    path: PathBuf,
    /// canonical sha → projected sha, or `None` if nothing of it is visible.
    entries: HashMap<String, Option<String>>,
    fresh: Vec<(String, Option<String>)>,
}

impl Map {
    fn load(git: &Git, remote: &str) -> Result<Map> {
        let path = git.sync_dir().join(format!("{remote}.map"));
        let mut entries = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if let Some((c, p)) = line.split_once(' ') {
                    entries.insert(c.to_string(), (p != "-").then(|| p.to_string()));
                }
            }
        }
        Ok(Map {
            path,
            entries,
            fresh: Vec::new(),
        })
    }

    fn get(&self, canonical: &str) -> Option<&Option<String>> {
        self.entries.get(canonical)
    }

    fn insert(&mut self, canonical: String, projected: Option<String>) {
        self.fresh.push((canonical.clone(), projected.clone()));
        self.entries.insert(canonical, projected);
    }

    fn projected(&self) -> impl Iterator<Item = &String> {
        self.entries.values().flatten()
    }

    /// Append-only: an entry, once written, is never changed.
    fn persist(&mut self) -> Result<()> {
        if self.fresh.is_empty() {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for (c, p) in self.fresh.drain(..) {
            writeln!(f, "{c} {}", p.as_deref().unwrap_or("-"))?;
        }
        Ok(())
    }
}

/// canonical sha → (remote, remote sha) for every replayed commit.
type Origins = HashMap<String, (String, String)>;

/// act key → canonical (sha, tree) candidates, parents before children. Several
/// commits made in the same second by the same person share a key.
type Acts = HashMap<String, Vec<(String, String)>>;

/// Who your own commits appear as on one side (an origin, or the canonical branch).
/// Identities that are not yours pass through untouched; dates are never changed,
/// so the mapping stays deterministic.
#[derive(Default)]
struct Persona {
    me: BTreeSet<String>,
    appear_as: Option<(String, String)>,
}

impl Persona {
    fn for_remote(profile: Option<&Profile>, remote: &Remote) -> Result<Persona> {
        let Some(p) = profile else {
            return match &remote.account {
                Some(id) => Err(TrackerError::Profile(format!(
                    "remote {} names account {id:?}, but there is no profile at {}",
                    remote.name,
                    Profile::path().display()
                ))),
                None => Ok(Persona::default()),
            };
        };
        let account = match &remote.account {
            Some(id) => Some(p.account(id).ok_or_else(|| {
                TrackerError::Profile(format!(
                    "remote {} names account {id:?}, which {} does not have",
                    remote.name,
                    p.path.display()
                ))
            })?),
            None => p.account_for_url(&remote.url),
        };
        Ok(Persona {
            me: p.me.clone(),
            appear_as: account.map(|a| (a.name.clone(), a.email.clone())),
        })
    }

    fn canonical(profile: Option<&Profile>) -> Persona {
        profile.map_or_else(Persona::default, |p| Persona {
            me: p.me.clone(),
            appear_as: Some((p.name.clone(), p.email.clone())),
        })
    }

    fn apply(&self, i: &Ident) -> Ident {
        match &self.appear_as {
            Some((name, email)) if self.me.contains(&i.email.to_lowercase()) => Ident {
                name: name.clone(),
                email: email.clone(),
                date: i.date.clone(),
            },
            _ => i.clone(),
        }
    }
}

/// A tree with every path hidden from one remote removed; memoised per tree.
#[derive(Default)]
struct TreeFilter {
    cache: HashMap<String, (String, bool)>,
}

impl TreeFilter {
    /// `(projected tree, projection is empty)`
    fn apply(&mut self, git: &Git, manifest: &Manifest, remote: &Remote, tree: &str) -> Result<(String, bool)> {
        if let Some(hit) = self.cache.get(tree) {
            return Ok(hit.clone());
        }
        let kept: Vec<_> = git
            .ls_tree(tree)?
            .into_iter()
            .filter(|e| manifest.visible(remote, &e.path))
            .collect();
        let out = (git.write_tree(&kept)?, kept.is_empty());
        self.cache.insert(tree.to_string(), out.clone());
        Ok(out)
    }
}

fn scan_origins(git: &Git, tip: &str) -> Result<Origins> {
    let mut out = HashMap::new();
    for (sha, body) in git.log_grep(tip, ORIGIN)? {
        if let Some(v) = trailer_value(&body, ORIGIN) {
            let mut it = v.split_whitespace();
            if let (Some(r), Some(s)) = (it.next(), it.next()) {
                out.insert(sha, (r.to_string(), s.to_string()));
            }
        }
    }
    Ok(out)
}

// ── entry point ──────────────────────────────────────────────────────────────

pub struct Options {
    pub dry_run: bool,
    pub pull: bool,
    pub push: bool,
    /// Restrict to these remotes; empty means all.
    pub only: Vec<String>,
    pub trust_messages: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Outcome {
    UpToDate,
    NothingVisible,
    Pushed { from: Option<String>, to: String },
    WouldPush { from: Option<String>, to: String },
}

#[derive(Debug, Default, Serialize)]
pub struct RemoteReport {
    pub name: String,
    pub replayed: Vec<String>,
    pub adopted: usize,
    pub created: usize,
    pub dropped: usize,
    pub outcome: Option<Outcome>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub branch: String,
    pub advanced: Option<(String, String)>,
    pub remotes: Vec<RemoteReport>,
    pub hook: Option<HookStatus>,
}

pub fn load_manifest(git: &Git) -> Result<Manifest> {
    let missing = || {
        TrackerError::Manifest(format!(
            "no {} committed at HEAD; run `tracker sync init`, edit it, and commit it",
            manifest::FILE
        ))
    };
    let at_head = Manifest::parse(&git.show_file("HEAD", manifest::FILE)?.ok_or_else(missing)?)?;
    // The canonical branch's own copy is authoritative.
    let branch_ref = format!("refs/heads/{}", at_head.branch);
    match git.show_file(&branch_ref, manifest::FILE)? {
        Some(text) => Manifest::parse(&text),
        None => Err(TrackerError::Manifest(format!(
            "branch {:?} has no {} committed",
            at_head.branch,
            manifest::FILE
        ))),
    }
}

pub fn sync(git: &Git, opts: &Options) -> Result<Report> {
    if git.sync_dir().join(RESOLVE_STATE).exists() {
        return Err(TrackerError::ResolveInProgress);
    }
    let manifest = load_manifest(git)?;
    let branch_ref = format!("refs/heads/{}", manifest.branch);
    let start = git.rev(&branch_ref)?.ok_or_else(|| {
        TrackerError::Manifest(format!("canonical branch {:?} does not exist", manifest.branch))
    })?;
    let remotes: Vec<&Remote> = if opts.only.is_empty() {
        manifest.remotes.iter().collect()
    } else {
        opts.only
            .iter()
            .map(|n| manifest.remote(n))
            .collect::<Result<_>>()?
    };
    let profile = Profile::load()?;
    let personas = remotes
        .iter()
        .map(|r| Persona::for_remote(profile.as_ref(), r))
        .collect::<Result<Vec<_>>>()?;
    let canonical = Persona::canonical(profile.as_ref());
    let hook = if opts.dry_run {
        None
    } else {
        Some(guard::install(git, &manifest)?)
    };

    let mut tips = Vec::new();
    for r in &remotes {
        tips.push(match git.remote_head(&r.url, &r.branch)? {
            None => None,
            Some(_) => {
                git.fetch(&r.url, &r.branch, &fetched_ref(&r.name))?;
                git.rev(&fetched_ref(&r.name))?
            }
        });
    }
    let mut maps = remotes
        .iter()
        .map(|r| Map::load(git, &r.name))
        .collect::<Result<Vec<_>>>()?;
    let mut reports: Vec<RemoteReport> = remotes
        .iter()
        .map(|r| RemoteReport {
            name: r.name.clone(),
            ..Default::default()
        })
        .collect();
    let mut origins = scan_origins(git, &start)?;
    let mut tip = start.clone();

    if opts.pull {
        for (i, r) in remotes.iter().enumerate() {
            let Some(remote_tip) = tips[i].clone() else { continue };
            let pulled = pull(
                git,
                &manifest,
                r,
                (&personas[i], &canonical),
                &remote_tip,
                &mut tip,
                &mut maps[i],
                &mut origins,
                &mut reports[i],
            );
            if let Err(e) = pulled {
                // Keep the replays that did go through, so a resolve starts from them.
                if !opts.dry_run {
                    if tip != start {
                        advance(git, &branch_ref, &start, &tip)?;
                    }
                    for m in &mut maps {
                        m.persist()?;
                    }
                }
                return Err(e);
            }
        }
    }

    let advanced = (tip != start).then(|| (start.clone(), tip.clone()));
    if advanced.is_some() && !opts.dry_run {
        advance(git, &branch_ref, &start, &tip)?;
    }

    if opts.push {
        let noted = git.noted(NOTES_REF)?;
        let order = git.rev_list_topo(&tip)?;
        for (i, r) in remotes.iter().enumerate() {
            let policy = if opts.trust_messages {
                MixedMessages::Keep
            } else {
                r.mixed_messages
            };
            let mut p = Projector {
                git,
                manifest: &manifest,
                remote: r,
                persona: &personas[i],
                policy,
                map: &mut maps[i],
                origins: &origins,
                noted: &noted,
                filter: TreeFilter::default(),
                trees: HashMap::new(),
                created: Vec::new(),
                dropped: 0,
                refused: Vec::new(),
            };
            for sha in &order {
                p.project(sha)?;
            }
            p.check()?;
            reports[i].created = p.created.len();
            reports[i].dropped = p.dropped;

            let remote_tip = tips[i].clone();
            let outcome = match maps[i].get(&tip).cloned().flatten() {
                None => Outcome::NothingVisible,
                Some(to) if remote_tip.as_deref() == Some(to.as_str()) => Outcome::UpToDate,
                Some(to) => {
                    if let Some(from) = &remote_tip {
                        if !git.is_ancestor(from, &to)? {
                            return Err(TrackerError::RemoteDiverged {
                                remote: r.name.clone(),
                            });
                        }
                    }
                    if opts.dry_run {
                        Outcome::WouldPush { from: remote_tip, to }
                    } else {
                        // The ref keeps the projected commits from being garbage-collected.
                        git.update_ref(&projected_ref(&r.name), &to, None)?;
                        git.push(&r.url, &to, &r.branch)?;
                        Outcome::Pushed { from: remote_tip, to }
                    }
                }
            };
            reports[i].outcome = Some(outcome);
            if !opts.dry_run {
                maps[i].persist()?;
            }
        }
    } else if !opts.dry_run {
        for m in &mut maps {
            m.persist()?;
        }
    }

    Ok(Report {
        branch: manifest.branch.clone(),
        advanced,
        remotes: reports,
        hook,
    })
}

/// Move the canonical branch forward, updating the working tree if it is checked out.
fn advance(git: &Git, branch_ref: &str, old: &str, new: &str) -> Result<()> {
    if git.head_branch()?.as_deref() == Some(branch_ref) {
        if git.is_dirty()? {
            return Err(TrackerError::DirtyWorktree);
        }
        git.run(&["merge", "--ff-only", "--quiet", new])?;
    } else {
        git.update_ref(branch_ref, new, Some(old))?;
    }
    Ok(())
}

// ── pull ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn pull(
    git: &Git,
    manifest: &Manifest,
    remote: &Remote,
    (persona, canonical): (&Persona, &Persona),
    remote_tip: &str,
    tip: &mut String,
    map: &mut Map,
    origins: &mut Origins,
    report: &mut RemoteReport,
) -> Result<()> {
    let known: Vec<String> = map
        .projected()
        .cloned()
        .chain(
            origins
                .values()
                .filter(|(r, _)| *r == remote.name)
                .map(|(_, s)| s.clone()),
        )
        .collect();
    let fresh = git.rev_list_new(remote_tip, known.iter())?;
    if fresh.is_empty() {
        return Ok(());
    }
    // Keyed by the identities this remote would have received, so earlier
    // projections are recognised even when they were given a per-host identity.
    let mut acts = Acts::new();
    for (sha, tree, author, committer) in git.idents(tip)? {
        let key = act_key(&persona.apply(&author), &persona.apply(&committer));
        acts.entry(key).or_default().push((sha, tree));
    }
    let mut filter = TreeFilter::default();

    for sha in fresh {
        let rc = git.commit(&sha)?;
        // One of our own projections whose map entry was lost: adopt it, don't replay
        // it. Identities alone can collide within a second, so the tree must match
        // too; the earliest unmapped candidate wins.
        let mut adopted = None;
        for (canonical, tree) in acts.get(&rc.act_key()).into_iter().flatten() {
            if map.get(canonical).is_none() && filter.apply(git, manifest, remote, tree)?.0 == rc.tree {
                adopted = Some(canonical.clone());
                break;
            }
        }
        if let Some(canonical) = adopted {
            map.insert(canonical, Some(sha));
            report.adopted += 1;
            continue;
        }
        let base = rc
            .parents
            .first()
            .ok_or_else(|| TrackerError::UnrelatedHistory {
                remote: remote.name.clone(),
                commit: sha.clone(),
            })?;
        for path in git.changed_paths(Some(base), &sha)? {
            if !manifest.visible(remote, &path) {
                return Err(TrackerError::HiddenFromRemote {
                    remote: remote.name.clone(),
                    commit: sha,
                    path,
                });
            }
        }
        let tree = match git.merge_tree(base, tip, &sha)? {
            Ok(tree) => tree,
            Err(paths) => {
                return Err(TrackerError::SyncConflict {
                    remote: remote.name.clone(),
                    commit: sha,
                    paths: paths.join(", "),
                })
            }
        };
        let message = with_origin(&rc.message, &remote.name, &sha);
        // Your own commits made on a host come home under your canonical identity.
        let replayed = git.commit_tree(
            &tree,
            std::slice::from_ref(tip),
            &canonical.apply(&rc.author),
            &canonical.apply(&rc.committer),
            &message,
        )?;
        origins.insert(replayed.clone(), (remote.name.clone(), sha.clone()));
        *tip = replayed;
        report.replayed.push(sha);
    }
    Ok(())
}

// ── push ─────────────────────────────────────────────────────────────────────

struct Projector<'a> {
    git: &'a Git,
    manifest: &'a Manifest,
    remote: &'a Remote,
    persona: &'a Persona,
    policy: MixedMessages,
    map: &'a mut Map,
    origins: &'a Origins,
    noted: &'a HashSet<String>,
    filter: TreeFilter,
    /// projected commit → its tree
    trees: HashMap<String, String>,
    created: Vec<String>,
    dropped: usize,
    /// (commit, subject) of mixed commits whose message was withheld.
    refused: Vec<(String, String)>,
}

impl Projector<'_> {
    fn tree_of(&mut self, commit: &str) -> Result<String> {
        if let Some(t) = self.trees.get(commit) {
            return Ok(t.clone());
        }
        let t = self.git.commit(commit)?.tree;
        self.trees.insert(commit.to_string(), t.clone());
        Ok(t)
    }

    /// Drop parents that are ancestors of other parents.
    fn reduce(&self, parents: Vec<String>) -> Result<Vec<String>> {
        if parents.len() < 2 {
            return Ok(parents);
        }
        let mut keep: Vec<String> = Vec::new();
        for (i, p) in parents.iter().enumerate() {
            let mut redundant = false;
            for (j, q) in parents.iter().enumerate() {
                if i != j && p != q && self.git.is_ancestor(p, q)? {
                    redundant = true;
                    break;
                }
            }
            if !redundant && !keep.contains(p) {
                keep.push(p.clone());
            }
        }
        Ok(keep)
    }

    fn project(&mut self, sha: &str) -> Result<()> {
        if self.map.get(sha).is_some() {
            return Ok(());
        }
        let c = self.git.commit(sha)?;
        let mut parents: Vec<String> = Vec::new();
        for p in &c.parents {
            match self.map.get(p) {
                Some(Some(pp)) if !parents.contains(pp) => parents.push(pp.clone()),
                Some(_) => {}
                None => {
                    return Err(TrackerError::Internal(format!(
                        "parent {p} of {sha} was not projected before it"
                    )))
                }
            }
        }
        let (tree, empty) = self.filter.apply(self.git, self.manifest, self.remote, &c.tree)?;

        let origin = self
            .origins
            .get(sha)
            .filter(|(r, _)| *r == self.remote.name)
            .map(|(_, s)| s.clone());
        if let Some(r) = &origin {
            let rc = self.git.commit(r)?;
            if rc.tree == tree && parents.len() == 1 && rc.parents.first() == parents.first() {
                // Nothing moved since the collaborator's commit: it IS the projection.
                self.trees.insert(r.clone(), rc.tree);
                self.map.insert(sha.to_string(), Some(r.clone()));
                return Ok(());
            }
            if !parents.contains(r) {
                parents.push(r.clone());
            }
        }
        let parents = self.reduce(parents)?;

        if origin.is_none() {
            let drop_to = match parents.as_slice() {
                [] if empty => Some(None),
                [only] if self.tree_of(only)? == tree => Some(Some(only.clone())),
                _ => None,
            };
            if let Some(target) = drop_to {
                self.map.insert(sha.to_string(), target);
                self.dropped += 1;
                return Ok(());
            }
        }

        let message = match &origin {
            Some(r) => format!("Merge {} commit {}\n", self.remote.name, short(r)),
            None => self.message(&c)?,
        };
        let new = self
            .git
            .commit_tree(
                &tree,
                &parents,
                &self.persona.apply(&c.author),
                &self.persona.apply(&c.committer),
                &message,
            )?;
        self.trees.insert(new.clone(), tree);
        self.map.insert(sha.to_string(), Some(new.clone()));
        self.created.push(new);
        Ok(())
    }

    fn override_for(&self, c: &Commit) -> Result<Option<String>> {
        let key = message_key(&self.remote.name);
        let pick = |text: &str| {
            trailer_value(text, &key)
                .or_else(|| trailer_value(text, SHARED_MESSAGE))
                .map(str::to_string)
        };
        if self.noted.contains(&c.sha) {
            if let Some(v) = self.git.note(NOTES_REF, &c.sha)?.as_deref().and_then(pick) {
                return Ok(Some(v));
            }
        }
        Ok(pick(&c.message))
    }

    fn message(&mut self, c: &Commit) -> Result<String> {
        if let Some(m) = self.override_for(c)? {
            return Ok(format!("{}\n", m.trim_end()));
        }
        let parent = c.parents.first().map(String::as_str);
        // The manifest is hidden everywhere, but saying a commit touched it gives nothing away.
        let touches_hidden = self
            .git
            .changed_paths(parent, &c.sha)?
            .iter()
            .any(|p| p != manifest::FILE && !self.manifest.visible(self.remote, p));
        if !touches_hidden {
            return Ok(clean_message(&c.message));
        }
        match self.policy {
            MixedMessages::Keep => Ok(clean_message(&c.message)),
            MixedMessages::Generic => Ok(GENERIC_MESSAGE.into()),
            MixedMessages::Refuse => {
                self.refused
                    .push((short(&c.sha).to_string(), c.subject().to_string()));
                Ok(GENERIC_MESSAGE.into())
            }
        }
    }

    /// Before anything is pushed: no withheld messages, and no hidden path in any
    /// tree this run created. The filter already guarantees the latter; this is the
    /// independent check that would catch it if it ever didn't.
    fn check(&mut self) -> Result<()> {
        if !self.refused.is_empty() {
            let listed = self
                .refused
                .iter()
                .take(8)
                .map(|(sha, subject)| format!("\n  {sha}  {subject}"))
                .collect::<String>();
            let more = self.refused.len().saturating_sub(8);
            return Err(TrackerError::MessageLeak {
                remote: self.remote.name.clone(),
                count: self.refused.len(),
                commits: if more > 0 {
                    format!("{listed}\n  … and {more} more")
                } else {
                    listed
                },
            });
        }
        let mut seen = HashSet::new();
        for c in &self.created {
            let tree = &self.trees[c];
            if !seen.insert(tree.clone()) {
                continue;
            }
            for e in self.git.ls_tree(tree)? {
                if !self.manifest.visible(self.remote, &e.path) {
                    return Err(TrackerError::LeakCheck {
                        remote: self.remote.name.clone(),
                        path: e.path,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_lines_never_survive_cleaning() {
        let m = "Fix intro\n\nMessage-uni: tidy\nShared-Message: x\nSync-Origin: uni abc\nSigned-off-by: A <a@b>\n";
        assert_eq!(clean_message(m), "Fix intro\n\nSigned-off-by: A <a@b>\n");
    }

    #[test]
    fn origin_joins_an_existing_trailer_block() {
        let m = "Fix intro\n\nSigned-off-by: A <a@b>\n";
        assert_eq!(
            with_origin(m, "uni", "abc"),
            "Fix intro\n\nSigned-off-by: A <a@b>\nSync-Origin: uni abc\n"
        );
        assert_eq!(with_origin("Fix: typo\n", "uni", "abc"), "Fix: typo\n\nSync-Origin: uni abc\n");
    }

    #[test]
    fn trailer_lookup() {
        let m = "Subject\n\nMessage-uni: for uni\nShared-Message: for all\n";
        assert_eq!(trailer_value(m, "Message-uni"), Some("for uni"));
        assert_eq!(trailer_value(m, SHARED_MESSAGE), Some("for all"));
        assert_eq!(trailer_value(m, "Message-co"), None);
    }
}
