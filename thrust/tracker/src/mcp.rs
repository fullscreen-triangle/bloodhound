//! `tracker mcp` — the operation registry served to AI agents over the Model
//! Context Protocol (JSON-RPC 2.0, one message per line on stdin/stdout).
//!
//! Every [`api`] operation is one tool. Questions are marked read-only; actions
//! that reach origins are marked open-world, so a client can ask the human before
//! running them. Nothing but protocol messages is ever written to stdout.

use crate::api::{self, Kind};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

/// Protocol revisions this server speaks, newest first.
const VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "tracker knows the user's repositories across every forge they use \
(GitHub, GitLab, Gitea), what each repository is about, and which paths of a repository each of \
its origins may see. Ask a question tool before an action tool. To change what an origin sees, \
use sync_hide / sync_unhide / sync_set_remote rather than editing .sync.toml by hand, then \
sync_status to preview and sync_run to apply. Errors carry a stable `code`: sync_conflict and \
hidden_from_remote are resolved with sync_resolve; message_leak with sync_message.";

fn tool(op: &api::Op) -> Value {
    let question = op.kind == Kind::Question;
    json!({
        "name": op.name,
        "description": op.summary,
        "inputSchema": op.schema(),
        "annotations": {
            "readOnlyHint": question,
            // Actions add or amend history and settings; none of them deletes work.
            "destructiveHint": false,
            "idempotentHint": question,
            "openWorldHint": op.network,
        },
    })
}

fn result(value: &Value) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_default();
    json!({ "content": [{ "type": "text", "text": text }], "structuredContent": value, "isError": false })
}

fn tool_error(e: &crate::error::TrackerError) -> Value {
    let body = json!({ "error": { "code": e.code(), "message": e.to_string() } });
    json!({ "content": [{ "type": "text", "text": body.to_string() }], "isError": true })
}

/// The response to one request, or `None` for a notification.
pub fn handle(msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    // Notifications (no id) never get a reply, whatever they are.
    let id = id?;

    let outcome: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
            let version = VERSIONS.iter().find(|v| **v == asked).unwrap_or(&VERSIONS[0]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "tracker", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": api::ops().iter().map(tool).collect::<Vec<_>>() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Value::Null);
            match api::ops().into_iter().find(|o| o.name == name) {
                None => Err((-32602, format!("unknown tool {name:?}"))),
                Some(op) => Ok(match op.call(args) {
                    Ok(v) => result(&v),
                    Err(e) => tool_error(&e),
                }),
            }
        }
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match outcome {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    })
}

/// Serve until stdin closes.
pub fn serve() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) if msg.is_object() => handle(&msg),
            Ok(_) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32600, "message": "expected a single JSON-RPC object" } })),
            Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } })),
        };
        if let Some(r) = reply {
            writeln!(stdout, "{r}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiates_a_known_version_and_lists_every_operation() {
        let r = handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}})).unwrap();
        assert_eq!(r["result"]["protocolVersion"], "2024-11-05");
        let r = handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})).unwrap();
        assert_eq!(r["result"]["protocolVersion"], VERSIONS[0]);

        let r = handle(&json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).unwrap();
        let tools = r["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), api::ops().len());
        let status = tools.iter().find(|t| t["name"] == "sync_status").unwrap();
        assert_eq!(status["annotations"]["readOnlyHint"], true);
        let run = tools.iter().find(|t| t["name"] == "sync_run").unwrap();
        assert_eq!(run["annotations"]["readOnlyHint"], false);
    }

    #[test]
    fn notifications_get_no_reply_and_failures_are_reported_in_band() {
        assert!(handle(&json!({"jsonrpc":"2.0","method":"notifications/initialized"})).is_none());
        let r = handle(&json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"sync_visibility","arguments":{"bogus":1}}})).unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("bad_args"));
        let r = handle(&json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"nope"}})).unwrap();
        assert_eq!(r["error"]["code"], -32602);
    }
}
