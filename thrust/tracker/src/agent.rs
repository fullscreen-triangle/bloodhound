//! A chat agent over the tracker's operations, run by a local Ollama model, Claude,
//! or a hosted model on Hugging Face (see `llm`).
//!
//! The model sees a chosen subset of [`crate::api`] operations as tools. Questions
//! run as soon as it asks. Actions never run here: each becomes a [`Proposal`] kept
//! in this process, and runs only when a person confirms it (see `serve`). The model
//! is told so, and is given the proposal ids to report back.
//!
//! Results that have a natural picture — recent commits, a repo's history, which
//! repos a query touches — come back with a chart spec the UI draws; the model can
//! also draw bar and pie charts itself with the `show_chart` tool.

use crate::api::{self, Kind};
use crate::error::{Result, TrackerError};
use crate::llm::{self, Call, ToolResult, ToolSpec, Turn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Operations the model may use. Everything else stays out of its reach.
const TOOLS: &[&str] = &[
    // questions
    "repos_recent", "graph_query", "repo_status", "git_read", "repo_sense",
    "repo_tree", "repo_file", "repo_diff", "repo_log", "repo_search",
    "sync_status", "sync_visibility", "sync_manifest", "tokens_status", "profile_repos",
    // actions — proposals only
    "repo_write", "repo_commit", "git_exec", "push_branch", "sync_run", "sync_hide", "sync_unhide", "sync_message",
    "codespace_open", "graph_build",
];
/// Model rounds per question: a small local model gets fewer, since each is slow.
const MAX_ROUNDS_LOCAL: usize = 5;
const MAX_ROUNDS_HOSTED: usize = 10;
/// What the model reads of one tool result. Every round re-reads the whole
/// conversation, and on a CPU-only Ollama that is what the wait is made of.
const TOOL_RESULT_CHARS: usize = 3000;
/// No new model round starts after this; what was found is returned as it is.
const TIME_BUDGET: Duration = Duration::from_secs(150);
// ── proposals ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    pub id: String,
    pub op: String,
    pub args: Value,
    /// One line a person can check before confirming, e.g. `git push github main  (in thrust)`.
    pub preview: String,
    pub summary: String,
    pub created: u64,
}

fn proposals() -> &'static Mutex<HashMap<String, Proposal>> {
    static P: OnceLock<Mutex<HashMap<String, Proposal>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn preview(op: &str, args: &Value) -> String {
    let repo = args["repo"].as_str().map(|r| format!("  (in {r})")).unwrap_or_default();
    let words = |v: &Value| {
        v.as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" "))
            .unwrap_or_default()
    };
    match op {
        "git_exec" => format!("git {}{repo}", words(&args["args"])),
        "push_branch" => format!(
            "git push <{} remote> {}{repo}",
            args["account"].as_str().unwrap_or("?"),
            args["branch"].as_str().unwrap_or("?")
        ),
        "sync_run" => format!("tracker sync run {}{repo}", words(&args["remotes"])),
        "sync_hide" => format!(
            "tracker sync hide {} --label {}{repo}",
            args["glob"].as_str().unwrap_or("?"),
            args["label"].as_str().unwrap_or("?")
        ),
        "sync_unhide" => format!("tracker sync unhide {}{repo}", args["glob"].as_str().unwrap_or("?")),
        "repo_write" if args["delete"] == true => format!("delete {}{repo}", args["path"].as_str().unwrap_or("?")),
        "repo_write" => format!(
            "write {} ({} bytes){repo}",
            args["path"].as_str().unwrap_or("?"),
            args["content"].as_str().map(str::len).unwrap_or(0)
        ),
        "repo_commit" => format!(
            "git commit -m {:?}{}{}{repo}",
            args["message"].as_str().unwrap_or(""),
            args["branch"].as_str().map(|b| format!(" on branch {b}")).unwrap_or_default(),
            if args["paths"].as_array().is_some_and(|p| !p.is_empty()) { format!(" — only {}", words(&args["paths"])) } else { " — all changes".into() }
        ),
        "codespace_open" => format!(
            "open a GitHub Codespace{}{repo}",
            args["branch"].as_str().map(|b| format!(" on {b}")).unwrap_or_default()
        ),
        _ => format!("{op} {}", args),
    }
}

/// Keep an action for a person to confirm. Checks the arguments against the
/// operation's schema first, so what is confirmed can actually run.
pub fn propose(op: &str, args: Value) -> Result<Proposal> {
    let o = api::ops()
        .into_iter()
        .find(|o| o.name == op)
        .ok_or_else(|| TrackerError::UnknownOperation(op.into()))?;
    if o.kind != Kind::Action {
        return Err(TrackerError::BadArgs(format!("{op} is a question; call it directly")));
    }
    check_args(&o.schema(), &args)?;
    let id = format!("p{:012x}", rand_u64() & 0xffff_ffff_ffff);
    let p = Proposal {
        id: id.clone(),
        op: op.into(),
        preview: preview(op, &args),
        summary: o.summary.into(),
        args,
        created: now(),
    };
    let mut all = proposals().lock().unwrap();
    all.retain(|_, p| now() - p.created < 24 * 3600);
    all.insert(id, p.clone());
    Ok(p)
}

/// Run a proposal a person confirmed. It is used up either way.
pub fn confirm(id: &str) -> Result<Value> {
    let p = proposals()
        .lock()
        .unwrap()
        .remove(id)
        .ok_or_else(|| TrackerError::BadArgs(format!("no pending proposal {id} (already run, rejected, or expired)")))?;
    let result = api::call(&p.op, p.args.clone())?;
    Ok(json!({ "proposal": p, "result": result, "charts": auto_charts(&p.op, &p.args, &result) }))
}

pub fn reject(id: &str) -> bool {
    proposals().lock().unwrap().remove(id).is_some()
}

pub fn pending() -> Vec<Proposal> {
    let mut v: Vec<Proposal> = proposals().lock().unwrap().values().cloned().collect();
    v.sort_by_key(|p| p.created);
    v
}

pub(crate) fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.write_u32(std::process::id());
    h.finish()
}

/// Required keys present, no keys the schema does not name.
fn check_args(schema: &Value, args: &Value) -> Result<()> {
    let empty = Map::new();
    let obj = match args {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err(TrackerError::BadArgs("arguments must be an object".into())),
    };
    let props = schema["properties"].as_object().cloned().unwrap_or_default();
    if let Some(k) = obj.keys().find(|k| !props.contains_key(*k)) {
        return Err(TrackerError::BadArgs(format!("unknown argument {k:?}")));
    }
    for r in schema["required"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        if !obj.contains_key(r) {
            return Err(TrackerError::BadArgs(format!("missing argument {r:?}")));
        }
    }
    Ok(())
}

// ── charts ───────────────────────────────────────────────────────────────────

/// Pictures for results that have an obvious one. Specs, not drawings: the UI draws.
pub fn auto_charts(op: &str, args: &Value, result: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    match op {
        "repos_recent" => {
            let items: Vec<Value> = result["repos"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|r| r["last_commit"].is_string())
                .map(|r| json!({ "label": r["name"], "date": r["last_commit"], "detail": r["last_subject"], "repo": r["name"] }))
                .collect();
            if !items.is_empty() {
                out.push(json!({ "type": "timeline", "title": "Last commit per repo", "items": items }));
            }
        }
        "repo_status" => {
            let items: Vec<Value> = result["commits"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|c| json!({ "label": c["subject"], "date": c["date"], "detail": format!("{} · {}", c["sha"].as_str().unwrap_or(""), c["author"].as_str().unwrap_or("")) }))
                .collect();
            if !items.is_empty() {
                let repo = args["repo"].as_str().unwrap_or("repo");
                out.push(json!({ "type": "timeline", "title": format!("Recent commits — {repo}"), "items": items }));
            }
        }
        "graph_query" => {
            let mut nodes: Vec<Value> = Vec::new();
            let mut links: Vec<Value> = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for r in result["repos"].as_array().into_iter().flatten() {
                let key = r["key"].as_str().unwrap_or("");
                if seen.insert(format!("r:{key}")) {
                    nodes.push(json!({ "id": format!("r:{key}"), "label": r["name"], "group": "repo" }));
                }
            }
            for (repo, triples) in result["evidence"].as_object().into_iter().flatten() {
                for t in triples.as_array().into_iter().flatten() {
                    let vid = format!("v:{}", t["value_id"].as_str().or(t["value"].as_str()).unwrap_or(""));
                    if seen.insert(vid.clone()) {
                        nodes.push(json!({ "id": vid, "label": t["value"], "group": "value" }));
                    }
                    links.push(json!({ "source": format!("r:{repo}"), "target": vid, "label": t["cue"] }));
                }
            }
            if nodes.len() > 1 {
                let q = args["query"].as_str().unwrap_or("");
                out.push(json!({ "type": "network", "title": format!("Repos about “{q}”"), "nodes": nodes, "links": links }));
            }
        }
        "repo_search" => {
            let items: Vec<Value> = result["results"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|p| json!({
                    "path": p["path"], "start": p["evidence_start_line"], "end": p["evidence_end_line"],
                    "snippet": p["snippet"], "matched": p["matched_terms"], "scene": p["scene"],
                }))
                .collect();
            out.push(json!({
                "type": "passages",
                "title": format!("“{}” in {}", args["query"].as_str().unwrap_or(""), args["repo"].as_str().unwrap_or("the repo")),
                "repo": args["repo"],
                "verdict": result["coverage"]["verdict"],
                "reason": result["coverage"]["reason"],
                "withheld": result["withheld"],
                "items": items,
            }));
        }
        "tokens_status" => {
            let rows: Vec<Value> = result
                .as_array()
                .or(result["tokens"].as_array())
                .into_iter()
                .flatten()
                .map(|t| json!([t["account"], t["state"], t["expires"], t["days_left"]]))
                .collect();
            if !rows.is_empty() {
                out.push(json!({ "type": "table", "title": "Tokens", "columns": ["account", "state", "expires", "days left"], "rows": rows }));
            }
        }
        _ => {}
    }
    out
}

/// A chart the model asked for with `show_chart`, checked and normalised.
fn model_chart(args: &Value) -> std::result::Result<Value, String> {
    let kind = args["type"].as_str().unwrap_or("bar");
    if kind != "bar" && kind != "pie" {
        return Err("type must be bar or pie".into());
    }
    let labels: Vec<String> = args["labels"]
        .as_array()
        .ok_or("labels must be a list of strings")?
        .iter()
        .map(|l| l.as_str().map(str::to_string).unwrap_or_else(|| l.to_string()))
        .collect();
    let values: Vec<f64> = args["values"]
        .as_array()
        .ok_or("values must be a list of numbers")?
        .iter()
        .map(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(f64::NAN))
        .collect();
    if labels.is_empty() || labels.len() != values.len() || values.iter().any(|v| !v.is_finite()) {
        return Err("labels and values must be equally long, with finite numbers".into());
    }
    let items: Vec<Value> = labels.iter().zip(&values).map(|(l, v)| json!({ "label": l, "value": v })).collect();
    Ok(json!({ "type": kind, "title": args["title"].as_str().unwrap_or(""), "items": items }))
}

// ── the loop ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub model: Option<String>,
    /// The repo the user is looking at, if any.
    pub focus: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Step {
    pub op: String,
    pub args: Value,
    pub kind: &'static str,
    pub ok: bool,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatReply {
    pub reply: String,
    pub model: String,
    pub steps: Vec<Step>,
    pub charts: Vec<Value>,
    pub proposals: Vec<Proposal>,
}

fn tool_specs() -> Vec<ToolSpec> {
    let mut specs: Vec<ToolSpec> = api::ops()
        .into_iter()
        .filter(|o| TOOLS.contains(&o.name))
        .map(|o| {
            let tag = if o.kind == Kind::Action { "ACTION (needs the user's confirmation): " } else { "" };
            ToolSpec { name: o.name.into(), description: format!("{tag}{}", o.summary), schema: o.schema() }
        })
        .collect();
    specs.push(ToolSpec {
        name: "show_chart".into(),
        description: "Draw a bar or pie chart for the user, e.g. files per language or commits per repo.".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "type": { "type": "string", "enum": ["bar", "pie"] },
                "title": { "type": "string" },
                "labels": { "type": "array", "items": { "type": "string" } },
                "values": { "type": "array", "items": { "type": "number" } },
            },
            "required": ["type", "labels", "values"],
        }),
    });
    specs
}

fn system_prompt(focus: Option<&str>) -> String {
    let repos: Vec<String> = crate::graph::load()
        .ok()
        .and_then(|g| g["repos"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|r| r["name"].as_str().map(str::to_string))
        .collect();
    let registered: Vec<String> = crate::registry::Federation::locate(&crate::registry::home_dir())
        .map(|f| f.repos.iter().map(|r| r.name.clone()).collect())
        .unwrap_or_default();
    let names = if registered.is_empty() { repos } else { registered };
    let accounts: Vec<String> = crate::profile::Profile::load()
        .ok()
        .flatten()
        .map(|p| p.accounts.iter().map(|a| format!("{} ({})", a.id, a.host)).collect())
        .unwrap_or_default();
    let mut s = String::from(
        "You are the tracker assistant. You answer questions about the user's git repositories and \
         prepare git work on them, using the tools. Rules:\n\
         - Get every fact from a tool. Never invent repo names, branches, commits, files or results.\n\
         - Pass repo names exactly as listed below in the `repo` argument.\n\
         - Tools marked ACTION do not run when you call them: each becomes a proposal the user confirms \
           in the interface. After proposing, say what you proposed and that it waits for confirmation. \
           Never claim an action was done.\n\
         - To push a branch to one forge account use push_branch. If the repo has a .sync.toml (several \
           origins with hidden paths), hide files from an origin with sync_hide and push with sync_run instead.\n\
         - Use git_read for read-only git (log, diff, show, ls-files); git_exec for anything that changes a repo.\n\
         - To find where a repo talks about something, call repo_search with the words the answer would \
           contain (keywords, not a question). Read coverage.verdict first: `declined` means the repo does \
           not contain those words — say \"this repo does not mention X\", try one other form of the word at \
           most, and never guess. `partial` means no passage holds all the words. Cite passages as \
           path:evidence_start_line-evidence_end_line and quote only the evidence shown.\n\
         - To compare numbers, call show_chart. Keep answers short and concrete.\n",
    );
    if !names.is_empty() {
        s.push_str(&format!("\nRepos: {}\n", names.join(", ")));
    }
    if !accounts.is_empty() {
        s.push_str(&format!("Forge accounts: {}\n", accounts.join(", ")));
    }
    if let Some(f) = focus {
        s.push_str(&format!("\nThe user is currently looking at the repo {f}.\n"));
    }
    s
}

/// Warm the default model if it is a local one, so the first question is not the
/// one that waits for Ollama to read the tool list.
pub fn warm() {
    let spec = llm::default_model();
    if !llm::is_local(&spec) {
        return;
    }
    let started = std::time::Instant::now();
    match llm::warm(&spec, &system_prompt(None), &tool_specs()) {
        Ok(()) => eprintln!("chat model {spec} ready ({}s)", started.elapsed().as_secs()),
        Err(e) => eprintln!("chat model {spec} not warmed ({e}); the first question will be slow"),
    }
}

/// Every model the chat can use, grouped by provider, with the default.
pub fn models() -> Result<Value> {
    Ok(llm::models())
}

/// Tool-call arguments as the operation expects them: nulls dropped, and a git
/// command given as one string split into words.
fn normalise(op: &str, raw: &Value) -> Value {
    let mut args = match raw {
        Value::Object(o) => o.clone(),
        Value::String(s) => serde_json::from_str::<Map<String, Value>>(s).unwrap_or_default(),
        _ => Map::new(),
    };
    args.retain(|_, v| !v.is_null() && v != "");
    // Small models send every value as a string ("10", "true", "[\"a\"]"): give each
    // the type the operation's schema asks for. A value that will not convert is
    // left alone, so the operation's own error explains it.
    let schema = api::ops().into_iter().find(|o| o.name == op).map(|o| o.schema());
    let props = schema.as_ref().and_then(|s| s["properties"].as_object().cloned()).unwrap_or_default();
    for (k, v) in args.iter_mut() {
        let Some(Value::String(text)) = Some(v.clone()) else { continue };
        let t = text.trim();
        let converted = match props.get(k).and_then(|p| p["type"].as_str()) {
            Some("integer") => t.parse::<i64>().ok().map(Value::from),
            Some("number") => t.parse::<f64>().ok().map(Value::from),
            Some("boolean") => match t.to_lowercase().as_str() {
                "true" | "yes" | "1" => Some(Value::Bool(true)),
                "false" | "no" | "0" => Some(Value::Bool(false)),
                _ => None,
            },
            Some("array") => serde_json::from_str::<Value>(t).ok().filter(Value::is_array).or_else(|| {
                let words = t.trim_start_matches("git ").split(|c: char| c.is_whitespace() || c == ',');
                Some(Value::Array(words.filter(|w| !w.is_empty()).map(|w| Value::String(w.into())).collect()))
            }),
            _ => None,
        };
        if let Some(c) = converted {
            *v = c;
        }
    }
    Value::Object(args)
}

/// A tool result cut down to what the model needs to answer; the full result
/// still feeds the charts.
fn compact(op: &str, r: &Value) -> Value {
    let take = |v: &Value, n: usize, f: &dyn Fn(&Value) -> Value| -> Value {
        Value::Array(v.as_array().into_iter().flatten().take(n).map(f).collect())
    };
    match op {
        "repos_recent" | "graph_query" => json!({
            "repos": take(&r["repos"], 20, &|x| json!({
                "name": x["name"], "last_commit": x["last_commit"], "last_subject": x["last_subject"],
                "hosts": take(&x["remotes"], 4, &|m| m["host"].clone()),
            })),
            "values": r["evidence"].as_object().map(|e| {
                e.iter().map(|(k, ts)| (k.clone(), take(ts, 4, &|t| t["value"].clone()))).collect::<Map<_, _>>()
            }),
        }),
        "repo_status" => json!({
            "branch": r["branch"], "branches": r["branches"], "remotes": r["remotes"],
            "uncommitted": r["changes"].as_array().map(Vec::len),
            "changes": take(&r["changes"], 15, &|c| c.clone()),
            "commits": take(&r["commits"], 10, &|c| json!(format!(
                "{} {} {} — {}",
                c["sha"].as_str().unwrap_or(""), c["date"].as_str().unwrap_or("").get(..10).unwrap_or(""),
                c["author"].as_str().unwrap_or(""), c["subject"].as_str().unwrap_or("")
            ))),
        }),
        "repo_log" => json!({ "commits": take(&r["commits"], 15, &|c| json!(format!(
            "{} {} {} — {}",
            c["short"].as_str().unwrap_or(""), c["date"].as_str().unwrap_or("").get(..10).unwrap_or(""),
            c["author"].as_str().unwrap_or(""), c["subject"].as_str().unwrap_or("")
        ))) }),
        "repo_search" => json!({
            "verdict": r["coverage"]["verdict"], "reason": r["coverage"]["reason"],
            "terms": r["coverage"]["terms"],
            "passages": take(&r["results"], 8, &|p| json!({
                "cite": format!("{}:{}-{}", p["path"].as_str().unwrap_or(""), p["evidence_start_line"], p["evidence_end_line"]),
                "matched": p["matched_terms"], "evidence": p["snippet"],
            })),
            "withheld_for_secrets": r["withheld"],
        }),
        "repo_tree" => json!({
            "files": r["files"].as_array().map(Vec::len),
            "changed": Value::Array(r["files"].as_array().into_iter().flatten().filter(|f| f["state"].is_string()).cloned().collect()),
            "paths": take(&r["files"], 150, &|f| f["path"].clone()),
        }),
        _ => r.clone(),
    }
}

fn clip(v: &Value) -> String {
    let s = v.to_string();
    if s.len() <= TOOL_RESULT_CHARS {
        return s;
    }
    let mut cut = TOOL_RESULT_CHARS;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}… (truncated, {} more characters)", &s[..cut], s.len() - cut)
}

/// History from the page as a conversation a model accepts: empty messages
/// dropped, and neighbours with the same role joined.
fn history(messages: &[Message]) -> Vec<Turn> {
    let mut out: Vec<(String, String)> = Vec::new();
    for m in messages {
        let text = m.content.trim();
        if text.is_empty() || !(m.role == "user" || m.role == "assistant") {
            continue;
        }
        match out.last_mut() {
            Some((role, prev)) if *role == m.role => {
                prev.push_str("\n\n");
                prev.push_str(text);
            }
            _ => out.push((m.role.clone(), text.to_string())),
        }
    }
    // A conversation starts with the user.
    while out.first().is_some_and(|(r, _)| r != "user") {
        out.remove(0);
    }
    out.into_iter()
        .map(|(role, text)| {
            if role == "user" {
                Turn::User(text)
            } else {
                Turn::Assistant { text, calls: Vec::new(), raw: Value::Null }
            }
        })
        .collect()
}

/// Run one tool call the model asked for: a question runs, an action becomes a
/// proposal, a chart is drawn.
fn run_call(c: &Call, steps: &mut Vec<Step>, charts: &mut Vec<Value>, made: &mut Vec<Proposal>) -> ToolResult {
    let name = c.name.clone();
    let args = normalise(&name, &c.args);
    let (content, is_error) = if name == "show_chart" {
        match model_chart(&args) {
            Ok(chart) => {
                charts.push(chart);
                ("The chart is shown to the user.".to_string(), false)
            }
            Err(e) => (format!("error: {e}"), true),
        }
    } else if !TOOLS.contains(&name.as_str()) {
        (format!("error: there is no tool {name:?}"), true)
    } else {
        let kind = api::ops().iter().find(|o| o.name == name).map(|o| o.kind);
        match kind {
            Some(Kind::Action) => match propose(&name, args.clone()) {
                Ok(p) => {
                    steps.push(Step { op: name.clone(), args, kind: "proposal", ok: true, error: None });
                    let text = format!(
                        "Proposed as {} — `{}`. It has NOT run; it waits for the user's confirmation.",
                        p.id, p.preview
                    );
                    made.push(p);
                    (text, false)
                }
                Err(e) => {
                    steps.push(Step { op: name.clone(), args, kind: "proposal", ok: false, error: Some(e.to_string()) });
                    (format!("error: {e}"), true)
                }
            },
            _ => match api::call(&name, args.clone()) {
                Ok(result) => {
                    charts.extend(auto_charts(&name, &args, &result));
                    steps.push(Step { op: name.clone(), args, kind: "question", ok: true, error: None });
                    (clip(&compact(&name, &result)), false)
                }
                Err(e) => {
                    steps.push(Step { op: name.clone(), args, kind: "question", ok: false, error: Some(e.to_string()) });
                    (
                        format!(
                            "FAILED — this call returned no data ({}): {e}. Fix the arguments and call again, or tell the user it failed.",
                            e.code()
                        ),
                        true,
                    )
                }
            },
        }
    };
    ToolResult { id: c.id.clone(), name, content, is_error }
}

pub fn chat(req: ChatRequest) -> Result<ChatReply> {
    let model = req.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(llm::default_model);
    let local = llm::is_local(&model);
    let max_rounds = if local { MAX_ROUNDS_LOCAL } else { MAX_ROUNDS_HOSTED };
    let tools = tool_specs();
    let system = system_prompt(req.focus.as_deref());
    let mut turns = history(&req.messages);
    if !matches!(turns.last(), Some(Turn::User(_))) {
        return Err(TrackerError::BadArgs("the conversation must end with a question from the user".into()));
    }
    let mut steps = Vec::new();
    let mut charts = Vec::new();
    let mut made = Vec::new();

    let started = std::time::Instant::now();
    for round in 0..max_rounds {
        if round > 0 && started.elapsed() > TIME_BUDGET {
            let ran: Vec<&str> = steps.iter().map(|s: &Step| s.op.as_str()).collect();
            return Ok(ChatReply {
                reply: format!(
                    "The model ran out of time before writing an answer. I ran {}; the results are below.",
                    ran.join(", ")
                ),
                model, steps, charts, proposals: made,
            });
        }
        // On the last round, no tools: the model must answer with what it has.
        let offered: &[ToolSpec] = if round + 1 < max_rounds { &tools } else { &[] };
        let reply = llm::complete(&model, &system, &turns, offered)?;
        if reply.calls.is_empty() {
            let mut text = reply.text;
            if made.is_empty() && text.to_lowercase().contains("propos") {
                text.push_str("\n\n(Note from the tracker: nothing was actually proposed in this turn — there is nothing to confirm.)");
            }
            return Ok(ChatReply { reply: text, model, steps, charts, proposals: made });
        }
        let results: Vec<ToolResult> = reply.calls.iter().map(|c| run_call(c, &mut steps, &mut charts, &mut made)).collect();
        turns.push(Turn::Assistant { text: reply.text, calls: reply.calls, raw: reply.raw });
        turns.push(Turn::Results(results));
    }
    Ok(ChatReply {
        reply: "I stopped after several tool calls without a final answer; the steps above show what I found.".into(),
        model,
        steps,
        charts,
        proposals: made,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_an_operation() {
        let names: Vec<&str> = api::ops().iter().map(|o| o.name).collect();
        for t in TOOLS {
            assert!(names.contains(t), "{t}");
        }
    }

    #[test]
    fn questions_are_not_proposed_and_bad_args_are_refused() {
        assert_eq!(propose("repo_status", json!({})).unwrap_err().code(), "bad_args");
        assert_eq!(propose("git_exec", json!({})).unwrap_err().code(), "bad_args");
        assert_eq!(propose("git_exec", json!({ "args": ["status"], "x": 1 })).unwrap_err().code(), "bad_args");
    }

    #[test]
    fn a_proposal_runs_once_and_only_when_confirmed() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git").args(["init", "-q"]).current_dir(dir.path()).status().unwrap();
        let p = propose("git_exec", json!({ "repo": dir.path(), "args": ["checkout", "-q", "-b", "feature"] })).unwrap();
        assert!(p.preview.starts_with("git checkout -q -b feature"));
        let branch = || {
            String::from_utf8(
                std::process::Command::new("git").args(["branch", "--show-current"]).current_dir(dir.path()).output().unwrap().stdout,
            )
            .unwrap()
        };
        assert_ne!(branch().trim(), "feature");
        confirm(&p.id).unwrap();
        assert_eq!(branch().trim(), "feature");
        assert!(confirm(&p.id).is_err());
    }

    #[test]
    fn model_charts_are_checked() {
        assert!(model_chart(&json!({ "type": "bar", "labels": ["a"], "values": [1] })).is_ok());
        assert!(model_chart(&json!({ "type": "bar", "labels": ["a", "b"], "values": [1] })).is_err());
        assert!(model_chart(&json!({ "type": "network", "labels": ["a"], "values": [1] })).is_err());
    }

    #[test]
    fn page_history_becomes_a_valid_conversation() {
        let m = |r: &str, c: &str| Message { role: r.into(), content: c.into() };
        let turns = history(&[m("assistant", "hi"), m("user", "a"), m("assistant", ""), m("user", "b"), m("assistant", "x")]);
        assert_eq!(turns.len(), 2);
        assert!(matches!(&turns[0], Turn::User(t) if t == "a\n\nb"));
        assert!(matches!(&turns[1], Turn::Assistant { text, .. } if text == "x"));
    }

    #[test]
    fn a_git_command_string_becomes_words() {
        let a = normalise("git_read", &json!({ "args": "git log --oneline -3", "repo": null }));
        assert_eq!(a, json!({ "args": ["log", "--oneline", "-3"] }));
    }

    #[test]
    fn string_values_get_their_schema_types() {
        let a = normalise("repos_recent", &json!({ "limit": "10" }));
        assert_eq!(a, json!({ "limit": 10 }));
        let a = normalise("repo_write", &json!({ "path": "a", "delete": "true" }));
        assert_eq!(a, json!({ "path": "a", "delete": true }));
        let a = normalise("repo_commit", &json!({ "message": "m", "paths": "[\"a\", \"b\"]" }));
        assert_eq!(a["paths"], json!(["a", "b"]));
    }
}
