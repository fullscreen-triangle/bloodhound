//! Everything `tracker` can answer or do, as one registry of named operations.
//!
//! Each operation is a *question* (reads, changes nothing that matters) or an
//! *action* (changes a repo, the federation, or a remote), takes a JSON object
//! described by a JSON Schema, and returns a JSON object. Every way into the tool
//! goes through here: `tracker call` / `tracker describe` for programs and scripts,
//! `tracker mcp` for AI agents, and [`call`] / [`ops`] for Rust. Handlers never
//! print, so their output is always machine-readable.

use crate::chi;
use crate::error::{Result, TrackerError};
use crate::{gitops, graph, search};
use crate::profile::{self, Profile};
use crate::purpose::{self, Index};
use crate::registry::Federation;
use crate::sync::edit::{self, RemoteSpec};
use crate::sync::engine::{self, Options};
use crate::sync::git::Git;
use crate::sync::manifest::MixedMessages;
use crate::sync::resolve;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Question,
    Action,
}

pub struct Op {
    pub name: &'static str,
    pub kind: Kind,
    /// Talks to forges or remotes over the network.
    pub network: bool,
    pub summary: &'static str,
    schema: fn() -> Value,
    run: fn(Value) -> Result<Value>,
}

impl Op {
    /// JSON Schema of the operation's arguments.
    pub fn schema(&self) -> Value {
        (self.schema)()
    }

    pub fn call(&self, args: Value) -> Result<Value> {
        (self.run)(args)
    }
}

/// Run the operation `name` with JSON `args` (an object, or null for none).
pub fn call(name: &str, args: Value) -> Result<Value> {
    ops()
        .into_iter()
        .find(|o| o.name == name)
        .ok_or_else(|| TrackerError::UnknownOperation(name.to_string()))?
        .call(args)
}

/// Every operation with its kind and argument schema: what a caller can ask for.
pub fn describe() -> Value {
    json!({
        "tool": "tracker",
        "version": env!("CARGO_PKG_VERSION"),
        "operations": ops().iter().map(|o| json!({
            "name": o.name,
            "kind": o.kind,
            "network": o.network,
            "summary": o.summary,
            "input_schema": o.schema(),
        })).collect::<Vec<_>>(),
    })
}

// ── argument plumbing ────────────────────────────────────────────────────────

fn args<T: DeserializeOwned>(v: Value) -> Result<T> {
    let v = if v.is_null() { json!({}) } else { v };
    serde_json::from_value(v).map_err(|e| TrackerError::BadArgs(e.to_string()))
}

fn to_value<T: Serialize>(t: &T) -> Result<Value> {
    serde_json::to_value(t).map_err(|e| TrackerError::Internal(format!("unserialisable result: {e}")))
}

/// A path to a repo, or the name of one tracked in the federation; default: cwd.
fn repo_dir(repo: Option<&str>) -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    match repo {
        None => Ok(cwd),
        Some(r) if Path::new(r).is_dir() => Ok(PathBuf::from(r)),
        Some(r) => Ok(Federation::locate(&cwd)?.get(r)?.path.clone()),
    }
}

fn open(repo: Option<&str>) -> Result<Git> {
    Git::open(&repo_dir(repo)?)
}

fn ensure_purpose() -> Result<()> {
    if purpose::is_available() {
        Ok(())
    } else {
        Err(TrackerError::PurposeMissing("`purpose --version` did not succeed".into()))
    }
}

const REPO: &str = "Repo directory, or the name of a repo tracked in the federation. Default: the current directory.";

fn schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn remotes_prop() -> Value {
    json!({ "type": "array", "items": { "type": "string" }, "description": "Only these remotes (default: all in the manifest)." })
}

// ── typed operations, also used by the CLI ───────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Drift {
    pub repo: String,
    pub previous: Option<f64>,
    pub current: f64,
    pub moved: bool,
    /// The repo's committed count after this act.
    pub committed: u64,
}

/// Recompute a tracked repo's χ, record it, and say whether its sense moved.
pub fn drift(repo: &str) -> Result<Drift> {
    let mut fed = Federation::locate(&std::env::current_dir()?)?;
    ensure_purpose()?;
    let path = fed.get(repo)?.path.clone();
    let previous = fed.get(repo)?.chi;
    purpose::index(&path)?;
    let current = chi::compute(&Index::load(&path, repo)?).chi;
    let committed = {
        let r = fed.get_mut(repo)?;
        r.record_act();
        r.chi = Some(current);
        r.committed
    };
    fed.save()?;
    Ok(Drift {
        repo: repo.to_string(),
        previous,
        current,
        moved: previous.is_some_and(|p| (current - p).abs() > 1e-9),
        committed,
    })
}

// ── the registry ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RepoArg {
    repo: Option<String>,
}

pub fn ops() -> Vec<Op> {
    vec![
        // ── federation: what each repo is about ──
        Op {
            name: "federation_list",
            kind: Kind::Question,
            network: false,
            summary: "List the repos tracked in the federation, with each one's character invariant χ and committed count.",
            schema: || schema(json!({}), &[]),
            run: |v| {
                let _: NoArgs = args(v)?;
                let fed = Federation::locate(&std::env::current_dir()?)?;
                Ok(json!({ "root": fed.root(), "repos": fed.repos }))
            },
        },
        Op {
            name: "repo_sense",
            kind: Kind::Question,
            network: false,
            summary: "What a repo is about. With a question: a fresh `purpose` search of it (file:line hits). \
                      Without: its standing sense — χ, load-bearing files, and the region its cheapest split severs. \
                      Refreshes the repo's .purpose index first.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "question": { "type": "string", "description": "A short stem to search for, e.g. \"parser\"; not a sentence." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, question: Option<String> }
                let a: A = args(v)?;
                let dir = repo_dir(a.repo.as_deref())?;
                let name = a.repo.unwrap_or_else(|| dir.display().to_string());
                ensure_purpose()?;
                purpose::index(&dir)?;
                match a.question {
                    Some(q) => {
                        let hits = purpose::ask(&dir, &q)?;
                        Ok(json!({ "repo": name, "question": q, "hits": hits }))
                    }
                    None => {
                        let c = chi::compute(&Index::load(&dir, &name)?);
                        Ok(json!({
                            "repo": name,
                            "chi": c.chi,
                            "blocks": c.blocks,
                            "core_blocks": c.core_blocks,
                            "fragments": c.fragments,
                            "salient": c.salient.iter().map(|(f, w)| json!({ "file": f, "weight": w })).collect::<Vec<_>>(),
                            "cut_side": c.cut_side,
                        }))
                    }
                }
            },
        },
        Op {
            name: "federation_ask",
            kind: Kind::Question,
            network: false,
            summary: "Search every tracked repo at once with `purpose`; hits per repo.",
            schema: || schema(json!({
                "question": { "type": "string", "description": "A short stem to search for, e.g. \"entropy\"." },
            }), &["question"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { question: String }
                let a: A = args(v)?;
                let fed = Federation::locate(&std::env::current_dir()?)?;
                ensure_purpose()?;
                let mut results = Vec::new();
                for r in &fed.repos {
                    purpose::index(&r.path)?;
                    let hits = purpose::ask(&r.path, &a.question)?;
                    if !hits.is_empty() {
                        results.push(json!({ "repo": r.name, "hits": hits }));
                    }
                }
                Ok(json!({ "question": a.question, "results": results }))
            },
        },
        Op {
            name: "repo_drift",
            kind: Kind::Action,
            network: false,
            summary: "Recompute a tracked repo's χ, record it (raising its committed count), and report whether its sense moved.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": "Name of a repo tracked in the federation." },
            }), &["repo"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: String }
                let a: A = args(v)?;
                to_value(&drift(&a.repo)?)
            },
        },
        // ── profile: you across every forge ──
        Op {
            name: "profile_show",
            kind: Kind::Question,
            network: false,
            summary: "The user's accounts on every forge and the name/email each host sees on their commits. Holds no secrets.",
            schema: || schema(json!({}), &[]),
            run: |v| {
                let _: NoArgs = args(v)?;
                to_value(&Profile::require()?)
            },
        },
        Op {
            name: "profile_repos",
            kind: Kind::Question,
            network: true,
            summary: "Every repository on the user's accounts (GitHub, GitLab, Gitea), copies of the same project grouped \
                      across hosts. Credentials come from git's credential helper (KeePassXC); `notes` says which accounts \
                      could only be listed partly.",
            schema: || schema(json!({
                "accounts": { "type": "array", "items": { "type": "string" }, "description": "Only these account ids (default: all)." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { #[serde(default)] accounts: Vec<String> }
                let a: A = args(v)?;
                to_value(&profile::repos(&a.accounts)?)
            },
        },
        // ── tokens: every account's token kept alive ──
        Op {
            name: "tokens_status",
            kind: Kind::Question,
            network: true,
            summary: "Check every account's token (held in KeePassXC): working, expiring soon, missing, or rejected, with \
                      its expiry date. Renews nothing. Tokens themselves are never returned.",
            schema: || schema(json!({
                "accounts": { "type": "array", "items": { "type": "string" }, "description": "Only these account ids (default: all)." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { #[serde(default)] accounts: Vec<String> }
                let a: A = args(v)?;
                let o = crate::tokens::Options { rotate: false, force: false, wait_unlock: false };
                Ok(json!({ "tokens": crate::tokens::refresh(&a.accounts, &o)? }))
            },
        },
        Op {
            name: "tokens_refresh",
            kind: Kind::Action,
            network: true,
            summary: "Renew every token that is due: GitLab tokens rotate themselves (the new token goes straight into \
                      KeePassXC, the old one is revoked); for GitHub and Gitea, which cannot rotate by API, it reports \
                      which to reissue by hand and where. Tokens themselves are never returned.",
            schema: || schema(json!({
                "accounts": { "type": "array", "items": { "type": "string" }, "description": "Only these account ids (default: all)." },
                "force": { "type": "boolean", "default": false, "description": "Rotate GitLab tokens even when not due." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { #[serde(default)] accounts: Vec<String>, #[serde(default)] force: bool }
                let a: A = args(v)?;
                let o = crate::tokens::Options { rotate: true, force: a.force, wait_unlock: false };
                Ok(json!({ "tokens": crate::tokens::refresh(&a.accounts, &o)? }))
            },
        },
        // ── sync: one repo, several origins, per-origin visibility ──
        Op {
            name: "sync_manifest",
            kind: Kind::Question,
            network: false,
            summary: "A repo's visibility manifest as committed on its canonical branch: its origins, what each sees, and the hide rules.",
            schema: || schema(json!({ "repo": { "type": "string", "description": REPO } }), &[]),
            run: |v| {
                let a: RepoArg = args(v)?;
                let m = engine::load_manifest(&open(a.repo.as_deref())?)?;
                Ok(json!({
                    "branch": m.branch,
                    "remotes": m.remotes,
                    "hide": m.hide_rules().into_iter().map(|(p, l)| json!({ "pattern": p, "label": l })).collect::<Vec<_>>(),
                }))
            },
        },
        Op {
            name: "sync_visibility",
            kind: Kind::Question,
            network: false,
            summary: "Which of a repo's origins may see each given path — e.g. \"can the university see bitspark/pricing.md?\"",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "paths": { "type": "array", "items": { "type": "string" }, "description": "Paths relative to the repo root, with /." },
            }), &["paths"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, paths: Vec<String> }
                let a: A = args(v)?;
                let m = engine::load_manifest(&open(a.repo.as_deref())?)?;
                let paths = a.paths.iter().map(|p| {
                    let p = p.replace('\\', "/");
                    let (vis, hid): (Vec<_>, Vec<_>) = m.remotes.iter().partition(|r| m.visible(r, &p));
                    json!({
                        "path": p,
                        "visible_to": vis.iter().map(|r| &r.name).collect::<Vec<_>>(),
                        "hidden_from": hid.iter().map(|r| &r.name).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>();
                Ok(json!({ "paths": paths }))
            },
        },
        Op {
            name: "sync_status",
            kind: Kind::Question,
            network: true,
            summary: "What a sync would do: collaborators' commits it would pull in, and what it would push to each origin. \
                      Fetches from the origins, but changes and pushes nothing.",
            schema: || schema(json!({ "repo": { "type": "string", "description": REPO }, "remotes": remotes_prop() }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, #[serde(default)] remotes: Vec<String> }
                let a: A = args(v)?;
                let report = engine::sync(&open(a.repo.as_deref())?, &Options {
                    dry_run: true, pull: true, push: true, only: a.remotes, trust_messages: false,
                })?;
                to_value(&report)
            },
        },
        Op {
            name: "sync_run",
            kind: Kind::Action,
            network: true,
            summary: "Pull collaborators' commits from every origin into the canonical branch, then push each origin the \
                      history it may see. Stops, changing nothing further, on conflicts, on collaborators writing into \
                      hidden paths, and on commit messages that might describe hidden work (error codes sync_conflict, \
                      hidden_from_remote, message_leak).",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "remotes": remotes_prop(),
                "pull": { "type": "boolean", "default": true, "description": "Take in collaborators' commits." },
                "push": { "type": "boolean", "default": true, "description": "Push each origin its projection." },
                "trust_messages": { "type": "boolean", "default": false, "description": "Keep original messages of commits that also touched hidden paths." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    repo: Option<String>,
                    #[serde(default)] remotes: Vec<String>,
                    #[serde(default = "yes")] pull: bool,
                    #[serde(default = "yes")] push: bool,
                    #[serde(default)] trust_messages: bool,
                }
                fn yes() -> bool { true }
                let a: A = args(v)?;
                let report = engine::sync(&open(a.repo.as_deref())?, &Options {
                    dry_run: false, pull: a.pull, push: a.push, only: a.remotes, trust_messages: a.trust_messages,
                })?;
                to_value(&report)
            },
        },
        Op {
            name: "sync_message",
            kind: Kind::Action,
            network: false,
            summary: "Set the one-line message an origin sees for a canonical commit that also touched paths hidden from it \
                      (answers a message_leak). Stored as a git note; history is not rewritten.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "commit": { "type": "string", "description": "The canonical commit (any rev)." },
                "text": { "type": "string", "description": "The single line the origin should see." },
                "remote": { "type": "string", "description": "The origin this message is for." },
                "shared": { "type": "boolean", "default": false, "description": "Use for every origin without its own message instead." },
            }), &["commit", "text"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, commit: String, text: String, remote: Option<String>, #[serde(default)] shared: bool }
                let a: A = args(v)?;
                let sha = crate::sync::set_message(&open(a.repo.as_deref())?, &a.commit, &a.text, a.remote.as_deref(), a.shared)?;
                Ok(json!({ "commit": sha, "remote": a.remote, "shared": a.shared, "text": a.text.trim() }))
            },
        },
        Op {
            name: "sync_resolve",
            kind: Kind::Action,
            network: true,
            summary: "Hand-resolve a blocked collaborator commit. `start` applies it, uncommitted, to the working tree and \
                      lists conflicted files; after fixing and `git add`, `continue` commits it; `abort` restores the tree.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "step": { "type": "string", "enum": ["start", "continue", "abort"] },
                "remote": { "type": "string", "description": "With start: the origin whose commit is blocked (default: first found)." },
            }), &["step"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, step: String, remote: Option<String> }
                let a: A = args(v)?;
                let git = open(a.repo.as_deref())?;
                match a.step.as_str() {
                    "start" => to_value(&resolve::start(&git, a.remote.as_deref())?),
                    "continue" => Ok(json!({ "committed": resolve::finish(&git)? })),
                    "abort" => { resolve::abort(&git)?; Ok(json!({ "aborted": true })) }
                    other => Err(TrackerError::BadArgs(format!("step must be start, continue or abort, not {other:?}"))),
                }
            },
        },
        Op {
            name: "sync_hide",
            kind: Kind::Action,
            network: false,
            summary: "Hide paths matching a glob from every origin that does not see `label` (e.g. pattern \"bitspark/\", \
                      label \"bitspark\"). Commits the manifest change on the canonical branch unless commit=false. \
                      Takes effect from the next sync; it cannot recall anything already pushed.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "pattern": { "type": "string", "description": "Glob from the repo root; a trailing / means the whole directory." },
                "label": { "type": "string", "description": "Only origins whose `sees` lists this label may see it. A label no origin sees never leaves." },
                "commit": { "type": "boolean", "default": true },
            }), &["pattern", "label"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, pattern: String, label: String, #[serde(default = "yes")] commit: bool }
                fn yes() -> bool { true }
                let a: A = args(v)?;
                to_value(&edit::hide(&open(a.repo.as_deref())?, &a.pattern, &a.label, a.commit)?)
            },
        },
        Op {
            name: "sync_unhide",
            kind: Kind::Action,
            network: false,
            summary: "Remove the hide rule for exactly this glob, so matching paths go to every origin from the next sync on.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "pattern": { "type": "string" },
                "commit": { "type": "boolean", "default": true },
            }), &["pattern"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, pattern: String, #[serde(default = "yes")] commit: bool }
                fn yes() -> bool { true }
                let a: A = args(v)?;
                to_value(&edit::unhide(&open(a.repo.as_deref())?, &a.pattern, a.commit)?)
            },
        },
        Op {
            name: "sync_set_remote",
            kind: Kind::Action,
            network: false,
            summary: "Add an origin to a repo's manifest, or change one: its URL and the labels it sees.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "name": { "type": "string", "description": "Origin name: letters, digits, - and _." },
                "url": { "type": "string" },
                "sees": { "type": "array", "items": { "type": "string" }, "description": "Labels this origin may see; [] means unlabelled paths only." },
                "branch": { "type": "string", "description": "Branch on the origin, if not the canonical branch's name." },
                "account": { "type": "string", "description": "Profile account whose identity this origin sees (default: by host)." },
                "mixed_messages": { "type": "string", "enum": ["refuse", "generic", "keep"] },
                "commit": { "type": "boolean", "default": true },
            }), &["name", "url", "sees"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    repo: Option<String>, name: String, url: String, sees: Vec<String>,
                    branch: Option<String>, account: Option<String>, mixed_messages: Option<MixedMessages>,
                    #[serde(default = "yes")] commit: bool,
                }
                fn yes() -> bool { true }
                let a: A = args(v)?;
                let spec = RemoteSpec {
                    name: &a.name, url: &a.url, sees: &a.sees, branch: a.branch.as_deref(),
                    account: a.account.as_deref(), mixed_messages: a.mixed_messages,
                };
                to_value(&edit::set_remote(&open(a.repo.as_deref())?, &spec, a.commit)?)
            },
        },
        // ── the federation graph (okgg) ──
        Op {
            name: "graph_build",
            kind: Kind::Action,
            network: false,
            summary: "Rebuild the federation knowledge graph: find every git repo under a root, describe each as a card, \
                      and let okgg individuate them into witnessed facets and values. Also registers every repo found \
                      in the home federation. The lexical generator takes seconds; ollama takes minutes.",
            schema: || schema(json!({
                "root": { "type": "string", "description": "Folder to scan (default: ~/Documents)." },
                "depth": { "type": "integer", "minimum": 1, "maximum": 8, "default": 4 },
                "generator": { "type": "string", "enum": ["lexical", "ollama"], "default": "lexical" },
                "model": { "type": "string", "description": "Ollama model, e.g. llama3.2." },
                "budget": { "type": "integer", "minimum": 1, "maximum": 200, "default": 20 },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { root: Option<PathBuf>, depth: Option<usize>, generator: Option<String>, model: Option<String>, budget: Option<u32> }
                let a: A = args(v)?;
                to_value(&graph::build(&graph::BuildOptions {
                    root: a.root.unwrap_or_else(graph::default_root),
                    depth: a.depth.unwrap_or(4),
                    generator: a.generator.unwrap_or_else(|| "lexical".into()),
                    model: a.model,
                    budget: a.budget.unwrap_or(20),
                })?)
            },
        },
        Op {
            name: "graph_query",
            kind: Kind::Question,
            network: false,
            summary: "Which repos are about something: repos whose name, facet value or witnessing cue word contains the query, \
                      with the triples that matched. Reads the last built graph.",
            schema: || schema(json!({
                "query": { "type": "string", "description": "One word or stem, e.g. \"entropy\" or \"rust\"." },
            }), &["query"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { query: String }
                let a: A = args(v)?;
                graph::query(&a.query)
            },
        },
        Op {
            name: "repos_recent",
            kind: Kind::Question,
            network: false,
            summary: "The repos committed to most recently, newest first, with branch, last commit subject, remotes and languages.",
            schema: || schema(json!({
                "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 15 },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { limit: Option<usize> }
                let a: A = args(v)?;
                graph::recent(a.limit.unwrap_or(15))
            },
        },
        // ── browsing and editing a repo ──
        Op {
            name: "repo_tree",
            kind: Kind::Question,
            network: false,
            summary: "Every file of a repo: in the working tree (with each changed file's git state), or at a branch, tag or commit.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "rev": { "type": "string", "description": "Branch, tag or commit; omit for the working tree." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, rev: Option<String> }
                let a: A = args(v)?;
                gitops::tree(&repo_dir(a.repo.as_deref())?, a.rev.as_deref())
            },
        },
        Op {
            name: "repo_file",
            kind: Kind::Question,
            network: false,
            summary: "One file's content, from the working tree or at a branch, tag or commit. Images come back base64-encoded.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "path": { "type": "string", "description": "Path inside the repo, with / separators." },
                "rev": { "type": "string", "description": "Branch, tag or commit; omit for the working tree." },
            }), &["path"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, path: String, rev: Option<String> }
                let a: A = args(v)?;
                gitops::file(&repo_dir(a.repo.as_deref())?, &a.path, a.rev.as_deref())
            },
        },
        Op {
            name: "repo_diff",
            kind: Kind::Question,
            network: false,
            summary: "A unified diff: the repo's uncommitted changes (optionally of one path), or what one commit changed.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "commit": { "type": "string", "description": "A commit to show; omit for uncommitted changes." },
                "path": { "type": "string" },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, commit: Option<String>, path: Option<String> }
                let a: A = args(v)?;
                gitops::diff(&repo_dir(a.repo.as_deref())?, a.commit.as_deref(), a.path.as_deref())
            },
        },
        Op {
            name: "repo_log",
            kind: Kind::Question,
            network: false,
            summary: "Commits of a branch (default: the current one), newest first, optionally only those touching a path.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "rev": { "type": "string" },
                "path": { "type": "string" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 50 },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, rev: Option<String>, path: Option<String>, limit: Option<usize> }
                let a: A = args(v)?;
                gitops::log(&repo_dir(a.repo.as_deref())?, a.rev.as_deref(), a.path.as_deref(), a.limit.unwrap_or(50))
            },
        },
        Op {
            name: "repo_write",
            kind: Kind::Action,
            network: false,
            summary: "Write a file in a repo's working tree (creating folders as needed), or delete it when `delete` is true. \
                      Paths must stay inside the repo and outside .git.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "path": { "type": "string" },
                "content": { "type": "string", "description": "The whole new content of the file." },
                "delete": { "type": "boolean", "default": false },
            }), &["path"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, path: String, content: Option<String>, #[serde(default)] delete: bool }
                let a: A = args(v)?;
                if !a.delete && a.content.is_none() {
                    return Err(TrackerError::BadArgs("give `content`, or `delete: true`".into()));
                }
                gitops::write(&repo_dir(a.repo.as_deref())?, &a.path, if a.delete { None } else { a.content.as_deref() })
            },
        },
        Op {
            name: "repo_commit",
            kind: Kind::Action,
            network: false,
            summary: "Commit changes in a repo: stage the given paths (default: every change) and commit with a message, \
                      first switching to `branch` (created if missing). Pushing is separate: push_branch, or sync_run for \
                      repos with a .sync.toml.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "message": { "type": "string" },
                "paths": { "type": "array", "items": { "type": "string" }, "description": "Only these paths (default: all changes)." },
                "branch": { "type": "string", "description": "Commit on this branch, created from the current commit if it does not exist." },
            }), &["message"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, message: String, #[serde(default)] paths: Vec<String>, branch: Option<String> }
                let a: A = args(v)?;
                gitops::commit(&repo_dir(a.repo.as_deref())?, &a.message, &a.paths, a.branch.as_deref())
            },
        },
        Op {
            name: "repo_search",
            kind: Kind::Question,
            network: false,
            summary: "Search inside a repo's files (spraypaint): ranked passages with evidence lines and a verdict on whether \
                      the repo covers the query at all — covered (one passage holds every word), partial, or declined (the \
                      words are not there). Write the words the answer would contain, not a question. Builds the search \
                      index on first use and after new commits.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "query": { "type": "string", "description": "Keywords the answer would contain, e.g. \"pairing token origin\"." },
                "k": { "type": "integer", "minimum": 1, "maximum": 50, "default": 8, "description": "Passages to return." },
                "scenes": { "type": "array", "items": { "type": "string" }, "description": "Only these top-level folders." },
                "commit": { "type": "boolean", "default": false, "description": "Record this search as a committed act (the count never goes down)." },
                "refresh": { "type": "boolean", "default": false, "description": "Rebuild the index first." },
            }), &["query"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    repo: Option<String>, query: String, k: Option<usize>,
                    #[serde(default)] scenes: Vec<String>, #[serde(default)] commit: bool, #[serde(default)] refresh: bool,
                }
                let a: A = args(v)?;
                let mut out = search::search(&repo_dir(a.repo.as_deref())?, &search::SearchOptions {
                    query: &a.query, k: a.k.unwrap_or(8), scenes: &a.scenes, commit: a.commit, refresh: a.refresh,
                })?;
                if let Some(r) = a.repo {
                    out["repo"] = json!(r);
                }
                Ok(out)
            },
        },
        // ── git on one repo ──
        Op {
            name: "repo_status",
            kind: Kind::Question,
            network: false,
            summary: "A repo's current branch, all branches, push remotes, uncommitted changes and last 20 commits.",
            schema: || schema(json!({ "repo": { "type": "string", "description": REPO } }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String> }
                let a: A = args(v)?;
                gitops::status(&repo_dir(a.repo.as_deref())?)
            },
        },
        Op {
            name: "git_read",
            kind: Kind::Question,
            network: false,
            summary: "Run a read-only git command in a repo (status, log, diff, show, branch, remote, ls-files, blame, grep, …). \
                      Commands that could change the repo are refused; use git_exec for those.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "args": { "type": "array", "items": { "type": "string" }, "description": "Arguments after `git`, e.g. [\"log\", \"--oneline\", \"-5\"]." },
            }), &["args"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, args: Vec<String> }
                let a: A = args(v)?;
                gitops::read(&repo_dir(a.repo.as_deref())?, &a.args)
            },
        },
        Op {
            name: "git_exec",
            kind: Kind::Action,
            network: true,
            summary: "Run any git command in a repo (commit, checkout, branch, rm --cached, push, …) and return its output. \
                      For a repo with several origins, prefer sync_run, which keeps hidden paths hidden.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "args": { "type": "array", "items": { "type": "string" }, "description": "Arguments after `git`." },
            }), &["args"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, args: Vec<String> }
                let a: A = args(v)?;
                gitops::exec(&repo_dir(a.repo.as_deref())?, &a.args)
            },
        },
        Op {
            name: "push_branch",
            kind: Kind::Action,
            network: true,
            summary: "Push a branch to the repo's remote on a profile account's forge (e.g. account \"github\"). \
                      Refused for repos with a .sync.toml — use sync_run there.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "branch": { "type": "string" },
                "account": { "type": "string", "description": "Profile account id; its host picks the remote." },
            }), &["branch", "account"]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, branch: String, account: String }
                let a: A = args(v)?;
                gitops::push_branch(&repo_dir(a.repo.as_deref())?, &a.branch, &a.account)
            },
        },
        Op {
            name: "codespace_open",
            kind: Kind::Action,
            network: true,
            summary: "Open a GitHub Codespace for a repo with a GitHub remote: reuse (and start) an existing one, or create one \
                      on the branch. Returns its web_url. Needs a GitHub token with the codespace scope.",
            schema: || schema(json!({
                "repo": { "type": "string", "description": REPO },
                "branch": { "type": "string", "description": "Branch to open (default: the repo's default branch)." },
            }), &[]),
            run: |v| {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A { repo: Option<String>, branch: Option<String> }
                let a: A = args(v)?;
                gitops::codespace(&repo_dir(a.repo.as_deref())?, a.branch.as_deref())
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_valid_tool_names() {
        let ops = ops();
        let mut names: Vec<_> = ops.iter().map(|o| o.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ops.len());
        for n in names {
            assert!(n.chars().all(|c| c.is_ascii_lowercase() || c == '_'), "{n}");
        }
    }

    #[test]
    fn every_schema_is_a_closed_object() {
        for o in ops() {
            let s = o.schema();
            assert_eq!(s["type"], "object", "{}", o.name);
            assert_eq!(s["additionalProperties"], false, "{}", o.name);
            for r in s["required"].as_array().unwrap() {
                assert!(s["properties"].get(r.as_str().unwrap()).is_some(), "{}: {r}", o.name);
            }
        }
    }

    #[test]
    fn unknown_operations_and_bad_arguments_are_errors_with_codes() {
        assert_eq!(call("nope", Value::Null).unwrap_err().code(), "unknown_operation");
        let err = call("sync_visibility", json!({ "paths": ["a"], "bogus": 1 })).unwrap_err();
        assert_eq!(err.code(), "bad_args");
    }
}
