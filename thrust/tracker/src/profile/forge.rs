//! Listing an account's repositories on GitHub, GitLab or Gitea.
//!
//! Tokens come from `git credential fill` (answered by KeePassXC once
//! `git-setup` has run) and reach `curl` on stdin, never on a command line.

use super::{Account, Forge};
use crate::error::{Result, TrackerError};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize)]
pub struct Copy {
    pub account: String,
    pub full_name: String,
    pub clone_url: String,
    pub ssh_url: String,
    pub web_url: String,
    pub private: bool,
    pub archived: bool,
    pub updated: String,
}

#[derive(Debug, Serialize)]
pub struct Project {
    /// Lowercased repository name, the key copies are grouped by.
    pub project: String,
    pub copies: Vec<Copy>,
}

pub struct Listing {
    pub repos: Vec<(String, Copy)>,
    pub authenticated: bool,
    /// git supplied a credential but the API rejected it.
    pub refused: bool,
}

/// Ask git's credential helpers for this host's secret, without ever prompting.
pub fn credential(host: &str, user: &str) -> Result<Option<String>> {
    let mut child = Command::new("git")
        .args(["credential", "fill"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GCM_INTERACTIVE", "never")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TrackerError::GitMissing(e.to_string()))?;
    let request = format!("protocol=https\nhost={host}\nusername={user}\n\n");
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(request.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Ok(None);
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("password="))
        .filter(|p| !p.is_empty())
        .map(str::to_string))
}

fn get_json(url: &str, headers: &[String]) -> Result<Value> {
    let mut child = Command::new("curl")
        .args(["-sS", "--fail-with-body", "-m", "30", "-L", "-H", "@-", url])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TrackerError::Forge(format!("could not run curl: {e}")))?;
    let mut input = String::from("Accept: application/json\nUser-Agent: tracker\n");
    for h in headers {
        input.push_str(h);
        input.push('\n');
    }
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(input.as_bytes())?;
    let out = child.wait_with_output()?;
    let body = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        let hint = if why.contains("401") || why.contains("403") {
            " — the credential was refused; KeePassXC should hold a personal access token for this host"
        } else {
            ""
        };
        return Err(TrackerError::Forge(format!(
            "{url}: {}{hint} {}",
            why.trim(),
            body.chars().take(200).collect::<String>()
        )));
    }
    serde_json::from_str(&body).map_err(|e| TrackerError::Forge(format!("{url}: not JSON: {e}")))
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn b(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// One API item → (name, copy), in each forge's own vocabulary.
fn parse(kind: Forge, account: &str, item: &Value) -> (String, Copy) {
    let (name, copy) = match kind {
        Forge::Github | Forge::Gitea => (
            s(item, "name"),
            Copy {
                account: account.to_string(),
                full_name: s(item, "full_name"),
                clone_url: s(item, "clone_url"),
                ssh_url: s(item, "ssh_url"),
                web_url: s(item, "html_url"),
                private: b(item, "private"),
                archived: b(item, "archived"),
                updated: if kind == Forge::Github {
                    s(item, "pushed_at")
                } else {
                    s(item, "updated_at")
                },
            },
        ),
        Forge::Gitlab => (
            s(item, "path"),
            Copy {
                account: account.to_string(),
                full_name: s(item, "path_with_namespace"),
                clone_url: s(item, "http_url_to_repo"),
                ssh_url: s(item, "ssh_url_to_repo"),
                web_url: s(item, "web_url"),
                // `simple` listings omit visibility; absent means we could not tell.
                private: item
                    .get("visibility")
                    .and_then(Value::as_str)
                    .is_some_and(|v| v != "public"),
                archived: b(item, "archived"),
                updated: s(item, "last_activity_at"),
            },
        ),
    };
    (name, copy)
}

fn page_url(a: &Account, authed: bool, page: u32) -> String {
    let (host, user) = (&a.host, &a.user);
    match (a.kind, authed) {
        (Forge::Github, _) => {
            let api = if host == "github.com" {
                "https://api.github.com".to_string()
            } else {
                format!("https://{host}/api/v3")
            };
            if authed {
                format!("{api}/user/repos?affiliation=owner,collaborator,organization_member&per_page=100&page={page}")
            } else {
                format!("{api}/users/{user}/repos?per_page=100&page={page}")
            }
        }
        (Forge::Gitlab, true) => {
            format!("https://{host}/api/v4/projects?membership=true&per_page=100&page={page}")
        }
        (Forge::Gitlab, false) => {
            format!("https://{host}/api/v4/users/{user}/projects?per_page=100&page={page}")
        }
        (Forge::Gitea, true) => format!("https://{host}/api/v1/user/repos?limit=50&page={page}"),
        (Forge::Gitea, false) => {
            format!("https://{host}/api/v1/users/{user}/repos?limit=50&page={page}")
        }
    }
}

fn fetch_all(a: &Account, token: Option<&str>) -> Result<Vec<(String, Copy)>> {
    let headers: Vec<String> = match (token, a.kind) {
        (None, _) => Vec::new(),
        (Some(t), Forge::Gitea) => vec![format!("Authorization: token {t}")],
        (Some(t), _) => vec![format!("Authorization: Bearer {t}")],
    };
    let mut repos = Vec::new();
    for page in 1..=50 {
        let v = get_json(&page_url(a, token.is_some(), page), &headers)?;
        let items = v.as_array().cloned().unwrap_or_default();
        if items.is_empty() {
            break;
        }
        repos.extend(items.iter().map(|item| parse(a.kind, &a.id, item)));
    }
    Ok(repos)
}

pub fn list(a: &Account) -> Result<Listing> {
    let token = credential(&a.host, &a.user)?;
    if let Some(t) = &token {
        match fetch_all(a, Some(t)) {
            Ok(repos) => return Ok(Listing { repos, refused: false, authenticated: true }),
            // A login password is not an API token; the public listing still helps.
            Err(TrackerError::Forge(msg)) if msg.contains("refused") => {
                return Ok(Listing {
                    repos: fetch_all(a, None).unwrap_or_default(),
                    refused: true,
                    authenticated: false,
                });
            }
            Err(e) => return Err(e),
        }
    }
    Ok(Listing {
        repos: fetch_all(a, None)?,
        refused: false,
        authenticated: false,
    })
}

/// Group copies by lowercased repository name.
pub fn group(repos: Vec<(String, Copy)>) -> Vec<Project> {
    let mut by: BTreeMap<String, Vec<Copy>> = BTreeMap::new();
    for (name, copy) in repos {
        by.entry(name.to_lowercase()).or_default().push(copy);
    }
    by.into_iter()
        .map(|(project, copies)| Project { project, copies })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_forge_is_read_in_its_own_vocabulary() {
        let gh = parse(
            Forge::Github,
            "github",
            &json!({"name":"Bloodhound","full_name":"ft/Bloodhound","private":false,
                    "clone_url":"https://github.com/ft/Bloodhound.git","ssh_url":"git@github.com:ft/Bloodhound.git",
                    "html_url":"https://github.com/ft/Bloodhound","pushed_at":"2026-09-01T00:00:00Z"}),
        );
        assert_eq!(gh.0, "Bloodhound");
        assert_eq!(gh.1.updated, "2026-09-01T00:00:00Z");

        let gl = parse(
            Forge::Gitlab,
            "bitspark",
            &json!({"path":"bloodhound","path_with_namespace":"you/bloodhound","visibility":"private",
                    "http_url_to_repo":"https://gitlab.example.com/you/bloodhound.git"}),
        );
        assert_eq!(gl.0, "bloodhound");
        assert!(gl.1.private);

        let gt = parse(
            Forge::Gitea,
            "uni",
            &json!({"name":"bloodhound","full_name":"you/bloodhound","private":true,"updated_at":"x"}),
        );
        assert!(gt.1.private);
        assert_eq!(gt.1.updated, "x");
    }

    #[test]
    fn copies_of_one_project_group_across_hosts() {
        let copy = |account: &str| Copy {
            account: account.into(),
            full_name: String::new(),
            clone_url: String::new(),
            ssh_url: String::new(),
            web_url: String::new(),
            private: false,
            archived: false,
            updated: String::new(),
        };
        let groups = group(vec![
            ("Bloodhound".into(), copy("github")),
            ("bloodhound".into(), copy("bitspark")),
            ("purpose".into(), copy("github")),
        ]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].project, "bloodhound");
        assert_eq!(groups[0].copies.len(), 2);
    }
}
