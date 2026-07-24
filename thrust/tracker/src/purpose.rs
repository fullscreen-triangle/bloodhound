//! Bridge to the `purpose` search CLI (the *search organ*).
//!
//! The tracker never re-implements search or indexing. It:
//!   * reads the self-graph that `purpose index` produced (`.purpose/index.json`), and
//!   * answers sense/where queries by *shelling out* to `purpose ask` — never from a
//!     stored answer (search-not-fetch, invariant I3).
//!
//! Index schema (confirmed against a real `.purpose/index.json`):
//! ```json
//! { "root": "<abs path>",
//!   "symbols": [ {"name","kind","file","line","snippet"}, ... ] }
//! ```

use crate::error::{Result, TrackerError};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One indexed definition/heading — a vertex of the repo self-graph.
#[derive(Debug, Clone, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    #[serde(default)]
    pub snippet: String,
}

/// The parsed `.purpose/index.json` — the repo's self-graph, owned by `purpose`.
#[derive(Debug, Clone, Deserialize)]
pub struct Index {
    #[allow(dead_code)]
    pub root: String,
    pub symbols: Vec<Symbol>,
}

impl Index {
    /// Path to the index cache inside a repo.
    pub fn path_in(repo_root: &Path) -> PathBuf {
        repo_root.join(".purpose").join("index.json")
    }

    /// Load and parse `.purpose/index.json` for a repo.
    pub fn load(repo_root: &Path, repo_name: &str) -> Result<Index> {
        let path = Self::path_in(repo_root);
        let bytes = std::fs::read(&path).map_err(|source| TrackerError::IndexUnreadable {
            repo: repo_name.to_string(),
            source,
        })?;
        // `purpose` writes UTF-8; be tolerant of a BOM.
        let text = String::from_utf8_lossy(&bytes);
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let index: Index =
            serde_json::from_str(text).map_err(|e| TrackerError::IndexMalformed {
                repo: repo_name.to_string(),
                line: e.line(),
                detail: e.to_string(),
            })?;
        Ok(index)
    }
}

/// Is the `purpose` binary reachable on PATH?
pub fn is_available() -> bool {
    Command::new("purpose")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `purpose index [--root <repo>]` to (re)build a repo's self-graph.
pub fn index(repo_root: &Path) -> Result<()> {
    run(&["index", "--root", &repo_root.to_string_lossy()]).map(|_| ())
}

/// One line of a `purpose ask` result slice: `file:line [kind] name` + snippet.
#[derive(Debug, Clone)]
pub struct AskHit {
    pub file: String,
    pub line: usize,
    pub kind: String,
    pub name: String,
    pub snippet: String,
}

/// Answer a query against a repo's *current* index via `purpose ask`.
///
/// This is the search-not-fetch organ (I3): the answer is a freshly computed slice,
/// never a stored value. `purpose`'s output is the ranked-slice text format; we parse
/// the `file:line  [kind] name` header lines and the indented snippet beneath each.
pub fn ask(repo_root: &Path, question: &str) -> Result<Vec<AskHit>> {
    let out = run(&["ask", question, "--root", &repo_root.to_string_lossy()])?;
    Ok(parse_ask(&out))
}

/// Parse the human-oriented slice `purpose ask` prints.
///
/// Example block:
/// ```text
/// 20 matching symbol(s):
///
///   backend/core/federated/base.py:32  [class] PrivacyEngine
///       class PrivacyEngine:
/// ```
fn parse_ask(text: &str) -> Vec<AskHit> {
    let mut hits = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim_start();
        // Header line looks like: `<file>:<line>  [<kind>] <name>`
        let Some((loc, rest)) = split_header(line) else {
            continue;
        };
        let Some((file, lineno)) = split_file_line(loc) else {
            continue;
        };
        let (kind, name) = split_kind_name(rest);
        // The next non-empty line, if it is more indented than a header, is the snippet.
        let snippet = match lines.peek() {
            Some(next) if is_snippet_line(next) => lines.next().unwrap().trim().to_string(),
            _ => String::new(),
        };
        hits.push(AskHit {
            file: file.to_string(),
            line: lineno,
            kind,
            name,
            snippet,
        });
    }
    hits
}

/// A header line carries a `[kind]` marker; snippet/prose lines do not begin with a loc.
fn split_header(line: &str) -> Option<(&str, &str)> {
    // Must contain "  [" (loc, then bracketed kind). Split on the first "  [".
    let idx = line.find("  [")?;
    let loc = line[..idx].trim();
    let rest = line[idx + 2..].trim(); // keep the leading '['
    if loc.contains(':') && rest.starts_with('[') {
        Some((loc, rest))
    } else {
        None
    }
}

fn split_file_line(loc: &str) -> Option<(&str, usize)> {
    let idx = loc.rfind(':')?;
    let (file, num) = loc.split_at(idx);
    let lineno = num[1..].trim().parse::<usize>().ok()?;
    Some((file, lineno))
}

fn split_kind_name(rest: &str) -> (String, String) {
    // rest is like "[class] PrivacyEngine"
    if let Some(end) = rest.find(']') {
        let kind = rest[1..end].trim().to_string();
        let name = rest[end + 1..].trim().to_string();
        (kind, name)
    } else {
        (String::new(), rest.trim().to_string())
    }
}

fn is_snippet_line(line: &str) -> bool {
    // Snippet lines are indented and are not themselves header lines.
    !line.trim().is_empty() && line.starts_with(' ') && split_header(line.trim_start()).is_none()
}

/// Low-level: run `purpose <args...>`, mapping absence and failure to typed errors.
fn run(args: &[&str]) -> Result<String> {
    let output = Command::new("purpose").args(args).output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            TrackerError::PurposeMissing(e.to_string())
        } else {
            TrackerError::Io(e)
        }
    })?;
    if !output.status.success() {
        return Err(TrackerError::PurposeFailed {
            args: args.join(" "),
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_slice() {
        let text = "\
20 matching symbol(s):

  backend/core/federated/base.py:32  [class] PrivacyEngine
      class PrivacyEngine:
  demo/demo.py:29  [def] demo_sequence_transformation
      def demo_sequence_transformation():
";
        let hits = parse_ask(text);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].file, "backend/core/federated/base.py");
        assert_eq!(hits[0].line, 32);
        assert_eq!(hits[0].kind, "class");
        assert_eq!(hits[0].name, "PrivacyEngine");
        assert_eq!(hits[0].snippet, "class PrivacyEngine:");
        assert_eq!(hits[1].name, "demo_sequence_transformation");
        assert_eq!(hits[1].snippet, "def demo_sequence_transformation():");
    }

    #[test]
    fn ignores_non_header_lines() {
        let text = "no matches here\njust prose\n";
        assert!(parse_ask(text).is_empty());
    }
}
