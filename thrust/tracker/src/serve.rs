//! `tracker serve`: the tracker as a local engine for the website.
//!
//! The site runs on Vercel but everything it shows lives on this machine — the repos,
//! the KeePassXC-held tokens, the Ollama model — so the browser calls this server on
//! the loopback address. Three locks keep other pages out:
//!
//! * **Host** must be the loopback name and port, which defeats DNS rebinding;
//! * **Origin**, when a browser sends one, must be one of the site's origins
//!   (`~/.tracker/serve.toml`, or the defaults below);
//! * every route but `/health` needs the **pairing token** as a Bearer header. The
//!   token lives in `~/.tracker/serve.token`; `tracker serve` prints a link that
//!   hands it to the site once, in the URL fragment, which never reaches a server.
//!
//! Questions run directly (`/call/{op}`). Actions only ever run from a proposal a
//! person confirmed (`/confirm/{id}`), whether the chat agent or a button made it.

use crate::agent;
use crate::api::{self, Kind};
use crate::error::{Result, TrackerError};
use crate::registry::home_dir;
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use tiny_http::{Header, Method, Request, Response, Server};

pub const DEFAULT_PORT: u16 = 8734;
const MAX_BODY: u64 = 1 << 20;
const DEFAULT_ORIGINS: &[&str] = &[
    "https://bloodhound-gamma.vercel.app",
    "https://st-hubert-bloodhound.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
];

pub struct Config {
    pub port: u16,
    pub origins: Vec<String>,
    /// The page the pairing link opens.
    pub site: String,
    pub token: String,
}

fn config_path() -> PathBuf {
    home_dir().join(".tracker").join("serve.toml")
}

fn token_path() -> PathBuf {
    home_dir().join(".tracker").join("serve.token")
}

/// The pairing token, made on first use. `renew` replaces it, unpairing every browser.
pub fn token(renew: bool) -> Result<String> {
    let path = token_path();
    if !renew {
        if let Ok(t) = std::fs::read_to_string(&path) {
            let t = t.trim().to_string();
            if t.len() >= 32 {
                return Ok(t);
            }
        }
    }
    let t: String = (0..4).map(|_| format!("{:016x}", agent::rand_u64())).collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, &t)?;
    Ok(t)
}

pub fn load_config(port: Option<u16>, extra: &[String], renew: bool) -> Result<Config> {
    let file: toml::Value = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(toml::Value::Table(Default::default()));
    let mut origins: Vec<String> = match file.get("origins").and_then(|o| o.as_array()) {
        Some(list) => list.iter().filter_map(|o| o.as_str().map(str::to_string)).collect(),
        None => DEFAULT_ORIGINS.iter().map(|s| s.to_string()).collect(),
    };
    origins.extend(extra.iter().cloned());
    for o in &mut origins {
        *o = o.trim_end_matches('/').to_string();
    }
    let site = file
        .get("site")
        .and_then(|s| s.as_str())
        .map(str::to_string)
        .or_else(|| origins.iter().find(|o| o.starts_with("https://")).cloned())
        .unwrap_or_else(|| "http://localhost:3000".into());
    let port = port
        .or_else(|| file.get("port").and_then(|p| p.as_integer()).map(|p| p as u16))
        .unwrap_or(DEFAULT_PORT);
    Ok(Config { port, origins, site: site.trim_end_matches('/').to_string(), token: token(renew)? })
}

/// Run until killed.
pub fn serve(cfg: Config) -> Result<()> {
    std::env::set_current_dir(home_dir())?;
    let addr = format!("127.0.0.1:{}", cfg.port);
    let server = Server::http(&addr).map_err(|e| TrackerError::Internal(format!("cannot listen on {addr}: {e}")))?;
    eprintln!("tracker engine on http://{addr}");
    eprintln!("allowed origins: {}", cfg.origins.join(", "));
    eprintln!("pair a browser (once) by opening:\n  {}/tracker#pair={}", cfg.site, cfg.token);
    std::thread::spawn(agent::warm);
    let cfg = std::sync::Arc::new(cfg);
    for req in server.incoming_requests() {
        let cfg = cfg.clone();
        std::thread::spawn(move || handle(&cfg, req));
    }
    Ok(())
}

fn header<'a>(req: &'a Request, name: &'static str) -> Option<&'a str> {
    req.headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str())
}

fn h(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn handle(cfg: &Config, mut req: Request) {
    let origin = header(&req, "Origin").map(str::to_string);
    let allowed = origin.as_deref().map(|o| cfg.origins.iter().any(|a| a == o));
    let host_ok = header(&req, "Host").is_some_and(|h| {
        h == format!("127.0.0.1:{}", cfg.port) || h == format!("localhost:{}", cfg.port)
    });

    let cors = |mut r: Response<std::io::Cursor<Vec<u8>>>| {
        if let (Some(o), Some(true)) = (&origin, allowed) {
            r.add_header(h("Access-Control-Allow-Origin", o));
            r.add_header(h("Vary", "Origin"));
        }
        r
    };
    let reply = |status: u16, body: Value| {
        cors(Response::from_data(body.to_string().into_bytes())
            .with_status_code(status)
            .with_header(h("Content-Type", "application/json"))
            .with_header(h("Cache-Control", "no-store")))
    };

    if !host_ok {
        let _ = req.respond(reply(421, json!({ "error": { "code": "bad_host", "message": "use 127.0.0.1" } })));
        return;
    }
    if allowed == Some(false) {
        let _ = req.respond(reply(403, json!({ "error": { "code": "origin", "message": "this origin is not allowed; add it to ~/.tracker/serve.toml" } })));
        return;
    }
    if *req.method() == Method::Options {
        let r = cors(Response::from_data(Vec::new()).with_status_code(204))
            .with_header(h("Access-Control-Allow-Methods", "GET, POST, OPTIONS"))
            .with_header(h("Access-Control-Allow-Headers", "authorization, content-type"))
            .with_header(h("Access-Control-Allow-Private-Network", "true"))
            .with_header(h("Access-Control-Max-Age", "600"));
        let _ = req.respond(r);
        return;
    }

    let url = req.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    let path = path.trim_end_matches('/').to_string();
    let bearer = header(&req, "Authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or("");
    let paired = same(bearer, &cfg.token);

    if path == "/health" {
        let _ = req.respond(reply(200, json!({
            "service": "bloodhound-engine",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": ["analyse", "tracker", "graph", "chat"],
            "paired": paired,
        })));
        return;
    }
    if !paired {
        let _ = req.respond(reply(401, json!({ "error": { "code": "unpaired", "message": "pair this browser: open the link `tracker serve` prints" } })));
        return;
    }

    let mut body = String::new();
    if *req.method() == Method::Post {
        let _ = req.as_reader().take(MAX_BODY).read_to_string(&mut body);
    }
    let body: Value = if body.trim().is_empty() { Value::Null } else { serde_json::from_str(&body).unwrap_or(Value::Null) };
    let method = req.method().clone();

    let result = route(&method, &path, query, body);
    let response = match result {
        Ok(v) => reply(200, v),
        Err(e) => {
            let status = match e.code() {
                "bad_args" | "unknown_operation" => 400,
                "not_found" => 404,
                _ => 500,
            };
            reply(status, json!({ "error": { "code": e.code(), "message": e.to_string() } }))
        }
    };
    let _ = req.respond(response);
}

fn param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| kv.strip_prefix(key)?.strip_prefix('='))
}

fn route(method: &Method, path: &str, query: &str, body: Value) -> Result<Value> {
    let get = *method == Method::Get;
    let post = *method == Method::Post;
    match path {
        "/describe" if get => Ok(api::describe()),
        "/graph" if get => crate::graph::load(),
        "/repos" if get => crate::graph::recent(param(query, "limit").and_then(|l| l.parse().ok()).unwrap_or(30)),
        "/models" if get => agent::models(),
        "/proposals" if get => Ok(json!({ "proposals": agent::pending() })),
        "/chat" if post => {
            let req: agent::ChatRequest = serde_json::from_value(body).map_err(|e| TrackerError::BadArgs(e.to_string()))?;
            Ok(serde_json::to_value(agent::chat(req)?).expect("serialisable"))
        }
        "/propose" if post => {
            let op = body["op"].as_str().ok_or_else(|| TrackerError::BadArgs("`op` is required".into()))?;
            Ok(json!({ "proposal": agent::propose(op, body["args"].clone())? }))
        }
        "/analyse" if post => analyse(&body),
        _ if post && path.starts_with("/call/") => {
            let op = &path["/call/".len()..];
            let o = api::ops().into_iter().find(|o| o.name == op).ok_or_else(|| TrackerError::UnknownOperation(op.into()))?;
            if o.kind == Kind::Action {
                return Err(TrackerError::BadArgs(format!("{op} is an action: POST /propose, then confirm it")));
            }
            let result = o.call(body.clone())?;
            let charts = agent::auto_charts(op, &body, &result);
            Ok(json!({ "result": result, "charts": charts }))
        }
        _ if post && path.starts_with("/confirm/") => agent::confirm(&path["/confirm/".len()..]),
        _ if post && path.starts_with("/reject/") => Ok(json!({ "rejected": agent::reject(&path["/reject/".len()..]) })),
        _ => Err(TrackerError::BadArgs(format!("no route {method} {path}"))),
    }
}

// ── /analyse: the Repo Lens local-repo snapshot ──────────────────────────────

const INDEXABLE: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp",
    "cs", "rb", "php", "swift", "kt", "scala", "md", "tex",
];
const IGNORED: &[&str] = &[
    ".git", "node_modules", "target", ".purpose", "dist", "build", "__pycache__",
    ".venv", "venv", ".next", ".nuxt", ".cache", "coverage", "out", "vendor", ".idea", ".vscode",
];

/// The files of a local repo, shaped like Repo Lens's GitHub fetch, so the page runs
/// its own χ on them. Tracked files only when it is a git repo.
fn analyse(body: &Value) -> Result<Value> {
    let asked = body["path"].as_str().ok_or_else(|| TrackerError::BadArgs("`path` is required".into()))?;
    let max_files = body["maxFiles"].as_u64().unwrap_or(400) as usize;
    let max_bytes = body["maxFileBytes"].as_u64().unwrap_or(120 * 1024);
    let dir = if Path::new(asked).is_dir() {
        PathBuf::from(asked)
    } else {
        crate::registry::Federation::locate(&home_dir())?.get(asked)?.path.clone()
    };
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(&dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let listed: Vec<String> = match git(&["ls-files"]) {
        Some(l) => l.lines().map(str::to_string).collect(),
        None => walk(&dir),
    };
    let mut files = Vec::new();
    let mut skipped = 0;
    let mut notes = Vec::new();
    for rel in listed {
        let ext = Path::new(&rel).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if !INDEXABLE.contains(&ext.as_str()) || rel.split('/').any(|c| IGNORED.contains(&c)) {
            continue;
        }
        let full = dir.join(&rel);
        let size = std::fs::metadata(&full).map(|m| m.len()).unwrap_or(0);
        if size > max_bytes || files.len() >= max_files {
            skipped += 1;
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&full) {
            files.push(json!({ "path": rel, "ext": ext, "text": text, "size": size }));
        }
    }
    if skipped > 0 {
        notes.push(format!("{skipped} files left out (over {max_files} files or {max_bytes} bytes each)"));
    }
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(json!({
        "mode": "snapshot",
        "repo": {
            "path": dir, "name": name,
            "commit": git(&["rev-parse", "HEAD"]),
            "defaultBranch": git(&["branch", "--show-current"]),
            "description": Value::Null, "language": Value::Null,
            "files": files, "skipped": skipped, "notes": notes,
        }
    }))
}

fn walk(root: &Path) -> Vec<String> {
    fn go(root: &Path, dir: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if IGNORED.contains(&name.as_str()) {
                continue;
            }
            if p.is_dir() {
                go(root, &p, out);
            } else if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = Vec::new();
    go(root, root, &mut out);
    out
}
