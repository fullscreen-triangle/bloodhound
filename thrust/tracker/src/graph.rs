//! The federation as a knowledge graph, built by `okgg`.
//!
//! Every local repository becomes one *card*: its README's opening, the names of its
//! top-level folders and files, its languages, and the descriptions its manifests
//! give. The cards are handed to `okgg`, which individuates them into witnessed
//! triples — each repo carries a value of a facet, backed by a cue word found in its
//! card — and reports what it could not tell apart as cells. The result, joined with
//! git facts (remotes, hosts, last commit) and the profile's accounts, is written to
//! `~/.tracker/graph/federation.json` for the dashboard and the chat agent.
//!
//! Building also registers every discovered repo in the home federation, so each one
//! can be named in `repo` arguments.

use crate::error::{Result, TrackerError};
use crate::profile::{host_of, Profile};
use crate::registry::{home_dir, Federation, RepoRecord};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const SKIP_DIRS: &[&str] = &["node_modules", "target", ".claude", "dist", "build", "__pycache__", "site-packages", ".venv", "venv"];
const CARD_README_WORDS: usize = 600;

fn graph_dir() -> PathBuf {
    home_dir().join(".tracker").join("graph")
}

pub fn graph_path() -> PathBuf {
    graph_dir().join("federation.json")
}

/// Where repos are looked for when no root is given: `~/Documents`.
pub fn default_root() -> PathBuf {
    home_dir().join("Documents")
}

#[derive(Debug, Clone, Serialize)]
pub struct Remote {
    pub name: String,
    pub url: String,
    pub host: Option<String>,
    /// The profile account whose host serves this remote.
    pub account: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Repo {
    /// Unique key: the path below the scan root, with `/` as separator.
    pub key: String,
    pub name: String,
    pub path: PathBuf,
    pub branch: Option<String>,
    pub last_commit: Option<String>,
    pub last_subject: Option<String>,
    pub commits: u64,
    pub remotes: Vec<Remote>,
    /// File extension → tracked file count, largest first (top 8).
    pub languages: Vec<(String, u64)>,
    pub files: u64,
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Every git work tree under `root`, not descending into another repo's folder.
pub fn discover(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, max: usize, out: &mut Vec<PathBuf>) {
        if dir.join(".git").exists() {
            out.push(dir.to_path_buf());
            return;
        }
        if depth >= max {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut subs: Vec<PathBuf> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                !name.starts_with('.') && !SKIP_DIRS.contains(&name)
            })
            .collect();
        subs.sort();
        for s in subs {
            walk(&s, depth + 1, max, out);
        }
    }
    let mut out = Vec::new();
    walk(root, 0, max_depth, &mut out);
    out
}

fn describe(root: &Path, dir: &Path, profile: Option<&Profile>) -> Repo {
    let key = dir
        .strip_prefix(root)
        .unwrap_or(dir)
        .to_string_lossy()
        .replace('\\', "/");
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut remotes: Vec<Remote> = Vec::new();
    for line in git(dir, &["remote", "-v"]).unwrap_or_default().lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(u)) = (it.next(), it.next()) {
            if remotes.iter().any(|r| r.name == n) {
                continue;
            }
            let host = host_of(u);
            let account = profile
                .and_then(|p| p.account_for_url(u))
                .map(|a| a.id.clone());
            remotes.push(Remote { name: n.into(), url: u.into(), host, account });
        }
    }
    let files_list = git(dir, &["ls-files"]).unwrap_or_default();
    let mut langs: BTreeMap<String, u64> = BTreeMap::new();
    let mut files = 0;
    for f in files_list.lines() {
        files += 1;
        if let Some(ext) = Path::new(f).extension().and_then(|e| e.to_str()) {
            *langs.entry(ext.to_lowercase()).or_default() += 1;
        }
    }
    let mut languages: Vec<(String, u64)> = langs.into_iter().collect();
    languages.sort_by(|a, b| b.1.cmp(&a.1));
    languages.truncate(8);
    let last = git(dir, &["log", "-1", "--format=%cI%x00%s"]).unwrap_or_default();
    let (last_commit, last_subject) = match last.split_once('\0') {
        Some((d, s)) => (Some(d.to_string()), Some(s.to_string())),
        None => (None, None),
    };
    Repo {
        key,
        name,
        path: dir.to_path_buf(),
        branch: git(dir, &["branch", "--show-current"]).filter(|b| !b.is_empty()),
        last_commit,
        last_subject,
        commits: git(dir, &["rev-list", "--count", "HEAD"]).and_then(|c| c.parse().ok()).unwrap_or(0),
        remotes,
        languages,
        files,
    }
}

/// The first `n` words of prose: badges, image links, URLs and HTML tags dropped,
/// since they say where a README's pictures live, not what the repo is.
fn first_words(text: &str, n: usize) -> String {
    let noise = |w: &&str| {
        let l = w.to_lowercase();
        l.contains("://")
            || l.starts_with("![")
            || l.starts_with('<')
            || l.contains('=')
            || l.ends_with('>')
            || l.contains("](")
            || l.starts_with("www.")
            || [".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"].iter().any(|e| l.contains(e))
    };
    text.split_whitespace()
        .filter(|w| !noise(w))
        .take(n)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The text okgg reads for one repo. Its name is left out on purpose: okgg tells
/// things apart by what they say, not by what they are called.
fn card(r: &Repo) -> String {
    let mut s = String::new();
    for readme in ["README.md", "readme.md", "README.rst", "README.txt", "README"] {
        if let Ok(t) = std::fs::read_to_string(r.path.join(readme)) {
            s.push_str(&first_words(&t, CARD_README_WORDS));
            s.push_str("\n\n");
            break;
        }
    }
    for manifest in ["Cargo.toml", "package.json", "pyproject.toml", "setup.cfg"] {
        if let Ok(t) = std::fs::read_to_string(r.path.join(manifest)) {
            for line in t.lines() {
                let l = line.trim();
                if l.starts_with("description") || l.starts_with("\"description\"") || l.starts_with("keywords") || l.starts_with("\"keywords\"") {
                    s.push_str(l);
                    s.push('\n');
                }
            }
        }
    }
    // Folder names say what a repo is made of; the ones every repo has say nothing.
    if let Ok(entries) = std::fs::read_dir(&r.path) {
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| {
                let l = n.to_lowercase();
                !n.starts_with('.') && !SKIP_DIRS.contains(&n.as_str()) && !COMMON_DIRS.contains(&l.as_str())
            })
            .collect();
        names.sort();
        if !names.is_empty() {
            s.push_str("\n## parts\n\n");
            s.push_str(&names.join(" "));
            s.push('\n');
        }
    }
    let langs: Vec<&str> = r.languages.iter().filter_map(|(e, _)| language(e)).collect();
    if !langs.is_empty() {
        s.push_str("\n## written in\n\n");
        s.push_str(&langs.join(" "));
        s.push('\n');
    }
    s
}

/// Folders nearly every repo has.
const COMMON_DIRS: &[&str] = &[
    "src", "lib", "docs", "doc", "test", "tests", "public", "assets", "static", "scripts", "images",
    "img", "figures", "examples", "bin", "config", "styles", "components", "pages", "utils", "data",
];

/// A file extension as the language it is written in; None for non-code files.
fn language(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "rs" => "rust",
        "py" | "ipynb" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" => "javascript",
        "go" => "go",
        "java" => "java",
        "kt" => "kotlin",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "scala" => "scala",
        "jl" => "julia",
        "r" => "r",
        "tex" | "bib" => "latex",
        "lean" => "lean",
        "hs" => "haskell",
        "ml" => "ocaml",
        "sql" => "sql",
        "sol" => "solidity",
        "wgsl" | "glsl" => "shaders",
        "sh" | "ps1" => "shell",
        _ => return None,
    })
}

#[derive(Debug, Serialize)]
pub struct Built {
    pub repos: usize,
    pub registered: usize,
    pub generator: String,
    pub value: Option<f64>,
    pub path: PathBuf,
}

pub struct BuildOptions {
    pub root: PathBuf,
    pub depth: usize,
    /// `ollama` (default in okgg) or `lexical` (seconds, no model).
    pub generator: String,
    pub model: Option<String>,
    pub budget: u32,
}

/// Discover, write the cards, run okgg, and save the joined graph.
pub fn build(o: &BuildOptions) -> Result<Built> {
    let profile = Profile::load()?;
    let found = discover(&o.root, o.depth);
    if found.len() < 2 {
        return Err(TrackerError::Profile(format!(
            "found {} git repositories under {}; need at least two",
            found.len(),
            o.root.display()
        )));
    }
    let repos: Vec<Repo> = found.iter().map(|d| describe(&o.root, d, profile.as_ref())).collect();

    // Cards, one file per repo. okgg shows its generator each file's name, so the
    // names are opaque (`r007.md`): a repo is told apart by what it says, never by
    // the folder it happens to sit in.
    let corpus = graph_dir().join("corpus");
    let _ = std::fs::remove_dir_all(&corpus);
    std::fs::create_dir_all(&corpus)?;
    let mut names = Names::new();
    for (i, r) in repos.iter().enumerate() {
        let card_name = format!("r{:03}.md", i + 1);
        std::fs::write(corpus.join(&card_name), card(r))?;
        names.insert(card_name, r.key.clone());
    }

    let out_dir = graph_dir().join("okgg");
    let mut cmd = Command::new("okgg");
    cmd.arg("run").arg(&corpus).arg("--out").arg(&out_dir)
        .args(["--generator", &o.generator])
        .args(["--budget", &o.budget.to_string()])
        .args(["--limit", "30"]);
    if let Some(m) = &o.model {
        cmd.args(["--model", m]);
    }
    let run = cmd
        .output()
        .map_err(|e| TrackerError::Forge(format!("could not run okgg: {e}; `cargo install --path conspirator/okgg`")))?;
    if !run.status.success() {
        return Err(TrackerError::Forge(format!(
            "okgg failed: {}",
            String::from_utf8_lossy(&run.stderr).lines().last().unwrap_or("")
        )));
    }
    let report: Value = serde_json::from_str(&std::fs::read_to_string(out_dir.join("report.json"))?)
        .map_err(|e| TrackerError::Forge(format!("okgg report unreadable: {e}")))?;

    let graph = join(&repos, &report, &o.generator, &names);
    std::fs::create_dir_all(graph_dir())?;
    std::fs::write(graph_path(), serde_json::to_string_pretty(&graph).expect("serialisable"))?;

    let registered = register(&repos)?;
    Ok(Built {
        repos: repos.len(),
        registered,
        generator: o.generator.clone(),
        value: report["value"].as_f64(),
        path: graph_path(),
    })
}

/// Card file name → repo key.
type Names = BTreeMap<String, String>;

fn join(repos: &[Repo], report: &Value, generator: &str, names: &Names) -> Value {
    // okgg keys entities by card file name; map them back to repo keys.
    let repo_key = |entity: &str| -> String {
        let e = entity.replace('\\', "/");
        names.get(&e).cloned().unwrap_or_else(|| e.strip_suffix(".md").unwrap_or(&e).to_string())
    };
    let keys = |v: &Value| -> Value {
        Value::Array(v.as_array().into_iter().flatten().filter_map(|k| k.as_str()).map(|k| Value::String(repo_key(k))).collect())
    };
    let status: BTreeMap<String, Value> = report["entities"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|e| {
            (
                repo_key(e["key"].as_str().unwrap_or("")),
                json!({ "status": e["status"], "open_pairs": e["open_pairs"], "indiscernible_from": keys(&e["indiscernible_from"]) }),
            )
        })
        .collect();
    let repo_values: Vec<Value> = repos
        .iter()
        .map(|r| {
            let mut v = serde_json::to_value(r).expect("serialisable");
            v["okgg"] = status.get(&r.key).cloned().unwrap_or(Value::Null);
            v
        })
        .collect();
    let facets: Vec<Value> = report["facets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|f| {
            json!({
                "id": f["id"],
                "property": f["property"],
                "gain": f["gain"],
                "values": f["values"].as_array().into_iter().flatten().map(|v| json!({
                    "id": v["id"], "name": v["name"], "cues": v["cues"], "carriers": v["carriers"],
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let links: Vec<Value> = report["triples"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            json!({
                "repo": repo_key(t["entity"].as_str().unwrap_or("")),
                "facet": t["facet"], "value": t["value"], "value_id": t["value_id"],
                "cue": t["cue"], "channel": t["channel"],
            })
        })
        .collect();
    let cells: Vec<Vec<String>> = report["cells"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            c.as_array()
                .into_iter()
                .flatten()
                .filter_map(|k| k.as_str().map(&repo_key))
                .collect()
        })
        .collect();
    let mut concepts = report["concepts"].clone();
    for c in concepts.as_array_mut().into_iter().flatten() {
        if let Some(ext) = c["extension"].as_array_mut() {
            for k in ext.iter_mut() {
                if let Some(s) = k.as_str() {
                    *k = Value::String(repo_key(s));
                }
            }
        }
    }
    let hosts: BTreeSet<String> = repos
        .iter()
        .flat_map(|r| r.remotes.iter().filter_map(|x| x.host.clone()))
        .collect();
    json!({
        "generated_at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "generator": generator,
        "okgg": {
            "value": report["value"], "stop": report["stop"], "n": report["n"],
            "certified": report["certified"], "open": report["open"], "indiscernible": report["indiscernible"],
            "trajectory": report["trajectory"], "concepts": concepts,
        },
        "hosts": hosts,
        "repos": repo_values,
        "facets": facets,
        "links": links,
        "cells": cells,
    })
}

/// Add every discovered repo to the home federation (without χ; `repo_drift` adds it).
fn register(repos: &[Repo]) -> Result<usize> {
    let home = home_dir();
    let mut fed = match Federation::locate(&home) {
        Ok(f) => f,
        Err(TrackerError::NoFederation(_)) => Federation::init(&home)?,
        Err(e) => return Err(e),
    };
    let mut added = 0;
    for r in repos {
        if fed.repos.iter().any(|x| x.path == r.path) {
            continue;
        }
        // Names must be unique; disambiguate by the key when a name repeats.
        let name = if fed.repos.iter().any(|x| x.name == r.name) {
            r.key.replace('/', "-")
        } else {
            r.name.clone()
        };
        fed.add(RepoRecord {
            name,
            path: r.path.clone(),
            remote: r.remotes.first().map(|x| x.url.clone()),
            chi: None,
            committed: 0,
        })?;
        added += 1;
    }
    fed.save()?;
    Ok(added)
}

/// The saved graph, if one has been built.
pub fn load() -> Result<Value> {
    let text = std::fs::read_to_string(graph_path()).map_err(|_| {
        TrackerError::Profile("no federation graph yet; run `tracker graph build`".into())
    })?;
    serde_json::from_str(&text).map_err(|e| TrackerError::Internal(format!("graph unreadable: {e}")))
}

/// Repos matching `query` by name, value name or cue, with the triples that match.
pub fn query(query: &str) -> Result<Value> {
    let g = load()?;
    let q = query.to_lowercase();
    let mut hits: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for l in g["links"].as_array().into_iter().flatten() {
        let value = l["value"].as_str().unwrap_or("").to_lowercase();
        let cue = l["cue"].as_str().unwrap_or("").to_lowercase();
        if value.contains(&q) || cue.contains(&q) {
            hits.entry(l["repo"].as_str().unwrap_or("").to_string()).or_default().push(l.clone());
        }
    }
    for r in g["repos"].as_array().into_iter().flatten() {
        let key = r["key"].as_str().unwrap_or("").to_string();
        if r["name"].as_str().unwrap_or("").to_lowercase().contains(&q) {
            hits.entry(key).or_default();
        }
    }
    let repos: Vec<Value> = g["repos"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| hits.contains_key(r["key"].as_str().unwrap_or("")))
        .cloned()
        .collect();
    Ok(json!({ "query": query, "repos": repos, "evidence": hits }))
}

/// Repos ordered by their last commit, newest first.
pub fn recent(limit: usize) -> Result<Value> {
    let g = load()?;
    let mut repos: Vec<Value> = g["repos"].as_array().cloned().unwrap_or_default();
    repos.sort_by(|a, b| b["last_commit"].as_str().unwrap_or("").cmp(a["last_commit"].as_str().unwrap_or("")));
    repos.truncate(limit);
    Ok(json!({ "repos": repos }))
}
