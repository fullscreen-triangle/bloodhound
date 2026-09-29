//! The machine interfaces, end to end: `tracker call` / `tracker describe` as a
//! script would use them, and `tracker mcp` as an AI agent's client would.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

struct Repo {
    _dir: tempfile::TempDir,
    root: PathBuf,
    canon: PathBuf,
    uni: PathBuf,
}

fn slash(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

impl Repo {
    fn env(&self, cmd: &mut Command) {
        cmd.env("GIT_CONFIG_GLOBAL", self.root.join("gitconfig"))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("TRACKER_PROFILE", self.root.join("no-profile.toml"))
            .env("GIT_AUTHOR_NAME", "Me")
            .env("GIT_AUTHOR_EMAIL", "me@example.org")
            .env("GIT_COMMITTER_NAME", "Me")
            .env("GIT_COMMITTER_EMAIL", "me@example.org");
    }

    fn git(&self, dir: &Path, args: &[&str]) -> String {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(dir).args(args);
        self.env(&mut cmd);
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn tracker(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tracker"));
        cmd.current_dir(&self.canon);
        self.env(&mut cmd);
        cmd
    }

    /// `tracker call`, returning (succeeded, parsed stdout).
    fn call(&self, op: &str, args: Value) -> (bool, Value) {
        let out = self.tracker().args(["call", op, &args.to_string()]).output().unwrap();
        let v = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|_| panic!("not JSON: {}", String::from_utf8_lossy(&out.stdout)));
        (out.status.success(), v)
    }
}

fn repo() -> Repo {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("gitconfig"), "").unwrap();
    let r = Repo {
        canon: root.join("canon"),
        uni: root.join("uni.git"),
        _dir: dir,
        root,
    };
    r.git(&r.root, &["init", "-q", "--bare", "-b", "main", &slash(&r.uni)]);
    r.git(&r.root, &["init", "-q", "-b", "main", &slash(&r.canon)]);
    std::fs::write(
        r.canon.join(".sync.toml"),
        format!("# my origins\nbranch = \"main\"\n\n[remotes.uni]\nurl = \"{}\"\nsees = []\n", slash(&r.uni)),
    )
    .unwrap();
    std::fs::write(r.canon.join("paper.md"), "paper\n").unwrap();
    std::fs::create_dir_all(r.canon.join("bitspark")).unwrap();
    std::fs::write(r.canon.join("bitspark/pricing.md"), "pricing\n").unwrap();
    r.git(&r.canon, &["add", "-A"]);
    r.git(&r.canon, &["commit", "-q", "-m", "Start"]);
    r
}

#[test]
fn describe_lists_questions_and_actions_with_schemas() {
    let r = repo();
    let out = r.tracker().arg("describe").output().unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    let ops = v["operations"].as_array().unwrap();
    let find = |n: &str| ops.iter().find(|o| o["name"] == n).unwrap().clone();
    assert_eq!(find("sync_visibility")["kind"], "question");
    assert_eq!(find("sync_hide")["kind"], "action");
    assert_eq!(find("sync_hide")["input_schema"]["required"], json!(["pattern", "label"]));
}

#[test]
fn a_script_can_ask_then_change_visibility_then_sync() {
    let r = repo();
    // Nothing hidden yet: the pricing file would reach the university.
    let (ok, v) = r.call("sync_visibility", json!({ "paths": ["bitspark/pricing.md"] }));
    assert!(ok, "{v}");
    assert_eq!(v["paths"][0]["visible_to"], json!(["uni"]));

    let (ok, v) = r.call("sync_hide", json!({ "pattern": "bitspark/", "label": "bitspark" }));
    assert!(ok, "{v}");
    assert_eq!(v["changed"], true);
    assert_eq!(r.git(&r.canon, &["log", "-1", "--format=%s"]), "tracker: hide bitspark/ as bitspark");
    // The edit kept the manifest's own comment.
    assert!(std::fs::read_to_string(r.canon.join(".sync.toml")).unwrap().starts_with("# my origins"));
    let (_, v) = r.call("sync_hide", json!({ "pattern": "bitspark/", "label": "bitspark" }));
    assert_eq!(v["changed"], false);

    let (_, v) = r.call("sync_visibility", json!({ "paths": ["bitspark/pricing.md", "paper.md"] }));
    assert_eq!(v["paths"][0]["hidden_from"], json!(["uni"]));
    assert_eq!(v["paths"][1]["visible_to"], json!(["uni"]));

    // The first commit touched both paper.md and bitspark/, so its message may
    // describe hidden work: the sync refuses, with a code the caller can act on.
    let (ok, v) = r.call("sync_run", json!({}));
    assert!(!ok);
    assert_eq!(v["error"]["code"], "message_leak");
    let (ok, v) = r.call("sync_message", json!({ "commit": "HEAD~1", "remote": "uni", "text": "Start the paper" }));
    assert!(ok, "{v}");

    let (ok, v) = r.call("sync_run", json!({}));
    assert!(ok, "{v}");
    assert_eq!(r.git(&r.uni, &["log", "--format=%s", "main"]), "Start the paper");
    assert_eq!(v["remotes"][0]["outcome"]["state"], "pushed");
    assert_eq!(r.git(&r.uni, &["ls-tree", "-r", "--name-only", "main"]), "paper.md");

    // A request the manifest cannot accept is refused, and nothing is written.
    let before = std::fs::read_to_string(r.canon.join(".sync.toml")).unwrap();
    let (ok, v) = r.call("sync_set_remote", json!({ "name": "bad name", "url": "x", "sees": [] }));
    assert!(!ok);
    assert_eq!(v["error"]["code"], "manifest");
    assert_eq!(std::fs::read_to_string(r.canon.join(".sync.toml")).unwrap(), before);
}

#[test]
fn failures_come_back_as_coded_json() {
    let r = repo();
    let (ok, v) = r.call("no_such_op", json!({}));
    assert!(!ok);
    assert_eq!(v["error"]["code"], "unknown_operation");
    let (_, v) = r.call("sync_visibility", json!({ "path": "typo" }));
    assert_eq!(v["error"]["code"], "bad_args");
}

#[test]
fn an_ai_client_can_drive_it_over_mcp() {
    let r = repo();
    let mut child = r
        .tracker()
        .args(["mcp", "--root", &slash(&r.canon)])
        .current_dir(&r.root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut send = |msg: Value| writeln!(stdin, "{msg}").unwrap();
    let mut recv = || {
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        serde_json::from_str::<Value>(&line).unwrap()
    };

    send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}));
    let init = recv();
    assert_eq!(init["result"]["serverInfo"]["name"], "tracker");
    send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));

    send(json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}));
    let list = recv();
    assert_eq!(list["id"], 2, "the notification must not have produced a reply");
    assert!(list["result"]["tools"].as_array().unwrap().iter().any(|t| t["name"] == "sync_hide"));

    send(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
        "name":"sync_hide","arguments":{"pattern":"bitspark/","label":"bitspark"}}}));
    assert_eq!(recv()["result"]["structuredContent"]["changed"], true);

    send(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
        "name":"sync_visibility","arguments":{"paths":["bitspark/pricing.md"]}}}));
    let vis = recv();
    assert_eq!(vis["result"]["structuredContent"]["paths"][0]["hidden_from"], json!(["uni"]));

    drop(stdin);
    assert!(child.wait().unwrap().success());
}
