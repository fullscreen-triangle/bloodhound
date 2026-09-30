//! The chat models the agent can think with, behind one interface.
//!
//! * `ollama:<model>` — a model on this machine; nothing leaves it.
//! * `claude:<model>` — Claude, through the Anthropic Messages API.
//! * `hf:<org/model>` — any model Hugging Face's Inference Providers serve with tool
//!   calling, through its OpenAI-compatible router; nothing is downloaded.
//! * `<name>:<model>` — any other OpenAI-compatible service listed in
//!   `~/.tracker/serve.toml` under `[[llm]]`.
//!
//! API keys live in KeePassXC, each in an entry whose URL is the API's host and whose
//! username is `api-key` (`tracker keys set <provider>` makes it), or in the
//! provider's environment variable. A key is checked for its provider's prefix before
//! it is sent anywhere, so a website password saved for the same host never is.

use crate::error::{Result, TrackerError};
use crate::registry::home_dir;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const KEY_USER: &str = "api-key";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Claude models offered in the model picker, most capable first.
const CLAUDE_MODELS: &[&str] = &["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-fable-5-1"];
/// Claude models that take the server-side refusal fallback.
const CLAUDE_FALLBACK: &[&str] = &["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"];
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

// ── a conversation any backend can replay ────────────────────────────────────

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

#[derive(Debug, Clone)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub args: Value,
}

pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

pub enum Turn {
    User(String),
    /// `raw` is the backend's own message, replayed unchanged (Claude's content
    /// blocks, thinking included); Null for history that came in as plain text.
    Assistant { text: String, calls: Vec<Call>, raw: Value },
    Results(Vec<ToolResult>),
}

pub struct Reply {
    pub text: String,
    pub calls: Vec<Call>,
    pub raw: Value,
    /// Why the model stopped, in the backend's words (`end_turn`, `tool_use`, `stop`, `refusal`, …).
    pub stop: String,
}

// ── providers ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Ollama,
    Anthropic,
    OpenAi,
}

#[derive(Debug, Clone, Serialize)]
pub struct Provider {
    pub name: String,
    pub kind: Kind,
    /// Who receives the conversation — shown to the user before they pick it.
    pub label: String,
    pub base: String,
    /// KeePassXC entry host holding the key (username `api-key`).
    pub key_host: Option<String>,
    pub key_env: Option<String>,
    #[serde(skip)]
    pub key_prefix: Option<String>,
    /// Where a key is made.
    pub key_url: Option<String>,
    pub models: Vec<String>,
}

fn config() -> toml::Value {
    std::fs::read_to_string(home_dir().join(".tracker").join("serve.toml"))
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(toml::Value::Table(Default::default()))
}

fn ollama_base() -> String {
    std::env::var("OLLAMA_HOST")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| if h.starts_with("http") { h } else { format!("http://{h}") })
        .unwrap_or_else(|| "http://127.0.0.1:11434".into())
        .trim_end_matches('/')
        .to_string()
}

pub fn providers() -> Vec<Provider> {
    let mut v = vec![
        Provider {
            name: "ollama".into(),
            kind: Kind::Ollama,
            label: "this machine (Ollama)".into(),
            base: ollama_base(),
            key_host: None,
            key_env: None,
            key_prefix: None,
            key_url: None,
            models: Vec::new(),
        },
        Provider {
            name: "claude".into(),
            kind: Kind::Anthropic,
            label: "Anthropic (Claude API)".into(),
            base: "https://api.anthropic.com".into(),
            key_host: Some("api.anthropic.com".into()),
            key_env: Some("ANTHROPIC_API_KEY".into()),
            key_prefix: Some("sk-ant-".into()),
            key_url: Some("https://console.anthropic.com/settings/keys".into()),
            models: CLAUDE_MODELS.iter().map(|m| m.to_string()).collect(),
        },
        Provider {
            name: "hf".into(),
            kind: Kind::OpenAi,
            label: "Hugging Face Inference Providers (and the provider serving the model)".into(),
            base: "https://router.huggingface.co/v1".into(),
            key_host: Some("router.huggingface.co".into()),
            key_env: Some("HF_TOKEN".into()),
            key_prefix: Some("hf_".into()),
            key_url: Some("https://huggingface.co/settings/tokens/new?tokenType=fineGrained".into()),
            models: Vec::new(),
        },
    ];
    // Extra OpenAI-compatible services:
    //   [[llm]]
    //   name = "openai"  base = "https://api.openai.com/v1"  key_host = "api.openai.com"
    //   key_prefix = "sk-"  models = ["gpt-5"]
    for e in config().get("llm").and_then(|l| l.as_array()).into_iter().flatten() {
        let s = |k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let (Some(name), Some(base)) = (s("name"), s("base")) else { continue };
        if v.iter().any(|p| p.name == name) {
            continue;
        }
        v.push(Provider {
            label: s("label").unwrap_or_else(|| name.clone()),
            name,
            kind: Kind::OpenAi,
            base: base.trim_end_matches('/').to_string(),
            key_host: s("key_host"),
            key_env: s("key_env"),
            key_prefix: s("key_prefix"),
            key_url: s("key_url"),
            models: e
                .get("models")
                .and_then(|m| m.as_array())
                .into_iter()
                .flatten()
                .filter_map(|m| m.as_str().map(str::to_string))
                .collect(),
        });
    }
    v
}

pub fn provider(name: &str) -> Result<Provider> {
    providers()
        .into_iter()
        .find(|p| p.name == name)
        .ok_or_else(|| TrackerError::BadArgs(format!("no chat provider {name:?} (known: ollama, claude, hf, or [[llm]] in serve.toml)")))
}

/// `provider:model`; a bare name is an Ollama model.
pub fn split(spec: &str) -> (String, String) {
    match spec.split_once(':') {
        Some((p, m)) if providers().iter().any(|x| x.name == p) => (p.to_string(), m.to_string()),
        _ => ("ollama".into(), spec.to_string()),
    }
}

/// The model used when a request names none: `TRACKER_MODEL`, then `model` in
/// serve.toml, then the local llama3.2.
pub fn default_model() -> String {
    std::env::var("TRACKER_MODEL")
        .ok()
        .filter(|m| !m.is_empty())
        .or_else(|| config().get("model").and_then(|m| m.as_str()).map(str::to_string))
        .unwrap_or_else(|| "ollama:llama3.2".into())
}

// ── keys ─────────────────────────────────────────────────────────────────────

fn key_cache() -> &'static Mutex<HashMap<String, String>> {
    static K: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    K.get_or_init(|| Mutex::new(HashMap::new()))
}

fn looks_right(p: &Provider, key: &str) -> bool {
    !key.trim().is_empty() && p.key_prefix.as_deref().map_or(true, |pre| key.starts_with(pre))
}

/// The provider's API key: from its environment variable, else from KeePassXC.
/// Kept in memory for the life of the process; never logged or returned.
pub fn key(p: &Provider) -> Result<String> {
    if let Some(k) = key_cache().lock().unwrap().get(&p.name) {
        return Ok(k.clone());
    }
    let missing = || {
        TrackerError::Keeper(format!(
            "no API key for {}: run `tracker keys set {}`{}",
            p.name,
            p.name,
            p.key_env.as_ref().map(|e| format!(" (or set {e})")).unwrap_or_default()
        ))
    };
    let from_env = p.key_env.as_ref().and_then(|e| std::env::var(e).ok()).filter(|k| !k.is_empty());
    let found = match from_env {
        Some(k) => Some(k),
        None => match &p.key_host {
            Some(host) => crate::tokens::keeper_secret(host, KEY_USER, true)?,
            None => return Err(missing()),
        },
    };
    let k = found.ok_or_else(missing)?;
    if !looks_right(p, &k) {
        return Err(TrackerError::Keeper(format!(
            "the key found for {} does not look like one (expected it to start with {:?}); \
             check the KeePassXC entry for {} with username {KEY_USER}",
            p.name,
            p.key_prefix.as_deref().unwrap_or(""),
            p.key_host.as_deref().unwrap_or("?")
        )));
    }
    key_cache().lock().unwrap().insert(p.name.clone(), k.clone());
    Ok(k)
}

/// Store a key for `provider` in KeePassXC, after checking it against the API.
pub fn set_key(provider_name: &str, key: &str) -> Result<String> {
    let p = provider(provider_name)?;
    let host = p
        .key_host
        .clone()
        .ok_or_else(|| TrackerError::BadArgs(format!("{} needs no key", p.name)))?;
    let key = key.trim();
    if !looks_right(&p, key) {
        return Err(TrackerError::BadArgs(format!(
            "that does not look like a {} key (expected it to start with {:?})",
            p.name,
            p.key_prefix.as_deref().unwrap_or("")
        )));
    }
    check_key(&p, key)?;
    crate::tokens::keeper_put(&host, KEY_USER, key)?;
    key_cache().lock().unwrap().insert(p.name.clone(), key.to_string());
    Ok(format!("stored in KeePassXC as {KEY_USER} @ https://{host}"))
}

/// Ask the provider whether it accepts the key, without spending tokens.
fn check_key(p: &Provider, key: &str) -> Result<()> {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(20)).build();
    let resp = match p.kind {
        Kind::Anthropic => agent
            .get(&format!("{}/v1/models", p.base))
            .set("x-api-key", key)
            .set("anthropic-version", ANTHROPIC_VERSION)
            .call(),
        Kind::OpenAi if p.name == "hf" => agent
            .get("https://huggingface.co/api/whoami-v2")
            .set("Authorization", &format!("Bearer {key}"))
            .call(),
        Kind::OpenAi => agent.get(&format!("{}/models", p.base)).set("Authorization", &format!("Bearer {key}")).call(),
        Kind::Ollama => return Ok(()),
    };
    match resp {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(401 | 403, _)) => Err(TrackerError::Forge(format!("{} rejected the key", p.label))),
        Err(ureq::Error::Status(code, _)) => Err(TrackerError::Forge(format!("{} answered {code} when checking the key", p.label))),
        Err(e) => Err(TrackerError::Forge(format!("could not reach {}: {e}", p.label))),
    }
}

/// Which providers have a key, without asking KeePassXC to show a dialog.
pub fn key_status() -> Vec<Value> {
    providers()
        .iter()
        .map(|p| {
            let state = match &p.key_host {
                None => "not needed".to_string(),
                Some(host) => {
                    let env = p.key_env.as_ref().and_then(|e| std::env::var(e).ok()).filter(|k| !k.is_empty());
                    match env {
                        Some(k) if looks_right(p, &k) => format!("from ${}", p.key_env.as_deref().unwrap_or("")),
                        Some(_) => "environment variable holds something else".into(),
                        None => match crate::tokens::keeper_secret(host, KEY_USER, false) {
                            Ok(Some(k)) if looks_right(p, &k) => "in KeePassXC".into(),
                            Ok(Some(_)) => "KeePassXC entry holds something that is not a key".into(),
                            Ok(None) => "missing".into(),
                            Err(e) => format!("unknown ({e})"),
                        },
                    }
                }
            };
            json!({ "provider": p.name, "label": p.label, "key": state, "entry": p.key_host.as_ref().map(|h| format!("https://{h}  (username {KEY_USER})")), "make_one_at": p.key_url })
        })
        .collect()
}

// ── the model list ───────────────────────────────────────────────────────────

fn hf_models() -> Vec<String> {
    static CACHE: OnceLock<Mutex<Option<(Instant, Vec<String>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if let Some((at, list)) = cache.lock().unwrap().as_ref() {
        if at.elapsed() < Duration::from_secs(3600) {
            return list.clone();
        }
    }
    let list: Vec<String> = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(10))
        .build()
        .get("https://router.huggingface.co/v1/models")
        .call()
        .ok()
        .and_then(|r| r.into_json::<Value>().ok())
        .map(|v| {
            v["data"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|m| {
                    m["providers"].as_array().into_iter().flatten().any(|p| {
                        p["supports_tools"] == true && p["status"].as_str().map_or(true, |s| s == "live")
                    })
                })
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if !list.is_empty() {
        *cache.lock().unwrap() = Some((Instant::now(), list.clone()));
    }
    list
}

fn ollama_models() -> Vec<String> {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(3))
        .build()
        .get(&format!("{}/api/tags", ollama_base()))
        .call()
        .ok()
        .and_then(|r| r.into_json::<Value>().ok())
        .map(|v| v["models"].as_array().into_iter().flatten().filter_map(|m| m["name"].as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Every model the picker may offer, grouped by provider.
pub fn models() -> Value {
    let groups: Vec<Value> = providers()
        .into_iter()
        .map(|p| {
            let models = match (p.kind, p.name.as_str()) {
                (Kind::Ollama, _) => ollama_models(),
                (_, "hf") => hf_models(),
                _ => p.models.clone(),
            };
            json!({
                "provider": p.name, "label": p.label, "local": p.kind == Kind::Ollama,
                "models": models.iter().map(|m| format!("{}:{m}", p.name)).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({ "default": default_model(), "providers": groups })
}

// ── one round with a model ───────────────────────────────────────────────────

fn http_error(p: &Provider, code: u16, body: &str) -> TrackerError {
    let detail: String = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v["error"]["message"].as_str().or(v["error"].as_str()).or(v["message"].as_str()).map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(300).collect());
    let hint = match code {
        401 | 403 => " — the key was rejected; store a new one with `tracker keys set`",
        402 => " — the account is out of credit",
        429 => " — rate limited; try again shortly",
        _ => "",
    };
    TrackerError::Forge(format!("{} answered {code}: {detail}{hint}", p.label))
}

fn post(p: &Provider, url: &str, headers: &[(&str, String)], body: &Value, timeout: u64) -> Result<Value> {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(timeout)).build();
    let mut last = None;
    for attempt in 0..2 {
        let mut req = agent.post(url);
        for (k, v) in headers {
            req = req.set(k, v);
        }
        match req.send_json(body.clone()) {
            Ok(r) => return r.into_json().map_err(|e| TrackerError::Internal(format!("{} reply unreadable: {e}", p.label))),
            // Overloaded or rate limited: one retry after a pause.
            Err(ureq::Error::Status(code @ (429 | 500 | 502 | 503 | 529), r)) if attempt == 0 => {
                last = Some(http_error(p, code, &r.into_string().unwrap_or_default()));
                std::thread::sleep(Duration::from_secs(3));
            }
            Err(ureq::Error::Status(code, r)) => return Err(http_error(p, code, &r.into_string().unwrap_or_default())),
            Err(e) if p.kind == Kind::Ollama => {
                return Err(TrackerError::Forge(format!("Ollama is not reachable at {} ({e}); start it with `ollama serve`", p.base)))
            }
            Err(e) => return Err(TrackerError::Forge(format!("could not reach {}: {e}", p.label))),
        }
    }
    Err(last.unwrap_or_else(|| TrackerError::Internal("no attempt made".into())))
}

fn function_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.schema } }))
        .collect()
}

/// Arguments sent as a JSON string (OpenAI style) or an object (Ollama style).
fn args_of(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Value::Object(_) => v.clone(),
        _ => json!({}),
    }
}

fn ollama(p: &Provider, model: &str, system: &str, turns: &[Turn], tools: &[ToolSpec], warm_only: bool) -> Result<Reply> {
    let mut messages = vec![json!({ "role": "system", "content": system })];
    for t in turns {
        match t {
            Turn::User(s) => messages.push(json!({ "role": "user", "content": s })),
            Turn::Assistant { raw, text, .. } if !raw.is_null() => {
                let _ = text;
                messages.push(raw.clone())
            }
            Turn::Assistant { text, .. } => messages.push(json!({ "role": "assistant", "content": text })),
            Turn::Results(rs) => {
                for r in rs {
                    messages.push(json!({ "role": "tool", "tool_name": r.name, "content": r.content }));
                }
            }
        }
    }
    let mut options = json!({ "temperature": 0.1, "num_ctx": 8192 });
    if warm_only {
        options["num_predict"] = json!(1);
    }
    let body = json!({
        "model": model, "messages": messages, "tools": function_tools(tools),
        "stream": false, "keep_alive": "30m", "options": options,
    });
    let v = post(p, &format!("{}/api/chat", p.base), &[], &body, 600)?;
    let msg = v["message"].clone();
    let calls = msg["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, c)| Call {
            id: format!("call_{i}"),
            name: c["function"]["name"].as_str().unwrap_or("").to_string(),
            args: args_of(&c["function"]["arguments"]),
        })
        .collect();
    Ok(Reply {
        text: msg["content"].as_str().unwrap_or("").trim().to_string(),
        calls,
        stop: v["done_reason"].as_str().unwrap_or("stop").to_string(),
        raw: msg,
    })
}

fn openai(p: &Provider, model: &str, system: &str, turns: &[Turn], tools: &[ToolSpec]) -> Result<Reply> {
    let key = key(p)?;
    let mut messages = vec![json!({ "role": "system", "content": system })];
    for t in turns {
        match t {
            Turn::User(s) => messages.push(json!({ "role": "user", "content": s })),
            Turn::Assistant { text, calls, .. } => {
                let mut m = Map::new();
                m.insert("role".into(), json!("assistant"));
                m.insert("content".into(), if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) });
                if !calls.is_empty() {
                    m.insert(
                        "tool_calls".into(),
                        Value::Array(
                            calls
                                .iter()
                                .map(|c| json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.args.to_string() } }))
                                .collect(),
                        ),
                    );
                }
                messages.push(Value::Object(m));
            }
            Turn::Results(rs) => {
                for r in rs {
                    messages.push(json!({ "role": "tool", "tool_call_id": r.id, "content": r.content }));
                }
            }
        }
    }
    let mut body = json!({
        "model": model, "messages": messages, "tool_choice": "auto",
        "max_tokens": 4096, "temperature": 0.2, "stream": false,
    });
    if !tools.is_empty() {
        body["tools"] = json!(function_tools(tools));
    }
    let v = post(p, &format!("{}/chat/completions", p.base), &[("Authorization", format!("Bearer {key}"))], &body, 180)?;
    let choice = &v["choices"][0];
    let msg = &choice["message"];
    let calls = msg["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, c)| Call {
            id: c["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("call_{i}")),
            name: c["function"]["name"].as_str().unwrap_or("").to_string(),
            args: args_of(&c["function"]["arguments"]),
        })
        .collect();
    Ok(Reply {
        text: msg["content"].as_str().unwrap_or("").trim().to_string(),
        calls,
        stop: choice["finish_reason"].as_str().unwrap_or("stop").to_string(),
        raw: msg.clone(),
    })
}

fn anthropic(p: &Provider, model: &str, system: &str, turns: &[Turn], tools: &[ToolSpec]) -> Result<Reply> {
    let key = key(p)?;
    let messages: Vec<Value> = turns
        .iter()
        .map(|t| match t {
            Turn::User(s) => json!({ "role": "user", "content": s }),
            // Claude's own content blocks go back exactly as they came.
            Turn::Assistant { raw, .. } if raw.is_array() => json!({ "role": "assistant", "content": raw }),
            Turn::Assistant { text, .. } => json!({ "role": "assistant", "content": text }),
            // All results of one turn in a single user message.
            Turn::Results(rs) => json!({ "role": "user", "content": rs.iter().map(|r| json!({
                "type": "tool_result", "tool_use_id": r.id, "content": r.content, "is_error": r.is_error,
            })).collect::<Vec<_>>() }),
        })
        .collect();
    let mut body = json!({
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": messages,
        "cache_control": { "type": "ephemeral" },
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.schema }))
            .collect::<Vec<_>>());
    }
    // Effort is not accepted by Haiku 4.5.
    if !model.starts_with("claude-haiku") {
        body["output_config"] = json!({ "effort": "medium" });
    }
    let mut betas: Vec<&str> = Vec::new();
    if CLAUDE_FALLBACK.contains(&model) {
        // A declined request is re-run server-side on the recommended model.
        body["fallbacks"] = json!("default");
        betas.push(FALLBACK_BETA);
    }
    let url = format!("{}/v1/messages", p.base);
    let headers = |betas: &[&str]| {
        let mut h = vec![("x-api-key", key.clone()), ("anthropic-version", ANTHROPIC_VERSION.to_string())];
        if !betas.is_empty() {
            h.push(("anthropic-beta", betas.join(",")));
        }
        h
    };
    let v = match post(p, &url, &headers(&betas), &body, 300) {
        Ok(v) => v,
        // If an optional feature is refused for this account or model, ask plainly.
        Err(TrackerError::Forge(m)) if m.contains("answered 400") && (body.get("fallbacks").is_some() || body.get("cache_control").is_some()) => {
            let map = body.as_object_mut().expect("object");
            map.remove("fallbacks");
            map.remove("cache_control");
            post(p, &url, &headers(&[]), &body, 300).map_err(|e| TrackerError::Forge(format!("{e} (first attempt: {m})")))?
        }
        Err(e) => return Err(e),
    };
    let content = v["content"].clone();
    let blocks = content.as_array().cloned().unwrap_or_default();
    let mut text: String = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    let calls = blocks
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .map(|b| Call {
            id: b["id"].as_str().unwrap_or("").to_string(),
            name: b["name"].as_str().unwrap_or("").to_string(),
            args: args_of(&b["input"]),
        })
        .collect();
    let stop = v["stop_reason"].as_str().unwrap_or("").to_string();
    match stop.as_str() {
        "refusal" => {
            let why = v["stop_details"]["explanation"].as_str().or(v["stop_details"]["category"].as_str()).unwrap_or("no reason given");
            text = format!("Claude declined this request ({why}).");
        }
        "max_tokens" => text.push_str("\n\n(The answer was cut off at the length limit.)"),
        _ => {}
    }
    Ok(Reply { text, calls, raw: content, stop })
}

/// One model round: the conversation so far in, the model's next message out.
pub fn complete(spec: &str, system: &str, turns: &[Turn], tools: &[ToolSpec]) -> Result<Reply> {
    let (name, model) = split(spec);
    let p = provider(&name)?;
    match p.kind {
        Kind::Ollama => ollama(&p, &model, system, turns, tools, false),
        Kind::Anthropic => anthropic(&p, &model, system, turns, tools),
        Kind::OpenAi => openai(&p, &model, system, turns, tools),
    }
}

/// Have a local model read the system prompt and tools once, so the first
/// question finds them in Ollama's prompt cache. Hosted models need no warming.
pub fn warm(spec: &str, system: &str, tools: &[ToolSpec]) -> Result<()> {
    let (name, model) = split(spec);
    let p = provider(&name)?;
    if p.kind == Kind::Ollama {
        ollama(&p, &model, system, &[Turn::User("ready?".into())], tools, true)?;
    }
    Ok(())
}

pub fn is_local(spec: &str) -> bool {
    provider(&split(spec).0).map(|p| p.kind == Kind::Ollama).unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_name_a_provider_or_default_to_ollama() {
        assert_eq!(split("claude:claude-opus-5-5"), ("claude".into(), "claude-opus-5-5".into()));
        assert_eq!(split("hf:Qwen/Qwen3.8-27B"), ("hf".into(), "Qwen/Qwen3.8-27B".into()));
        assert_eq!(split("ollama:llama3.2:latest"), ("ollama".into(), "llama3.2:latest".into()));
        assert_eq!(split("llama3.2:latest"), ("ollama".into(), "llama3.2:latest".into()));
    }

    #[test]
    fn a_website_password_is_not_taken_for_a_key() {
        let claude = provider("claude").unwrap();
        assert!(looks_right(&claude, "sk-ant-api03-abc"));
        assert!(!looks_right(&claude, "hunter2"));
        let hf = provider("hf").unwrap();
        assert!(looks_right(&hf, "hf_abc"));
        assert!(!looks_right(&hf, "sk-ant-api03-abc"));
    }

    #[test]
    fn arguments_come_as_strings_or_objects() {
        assert_eq!(args_of(&json!("{\"a\":1}")), json!({ "a": 1 }));
        assert_eq!(args_of(&json!({ "a": 1 })), json!({ "a": 1 }));
        assert_eq!(args_of(&json!("not json")), json!({}));
    }
}
