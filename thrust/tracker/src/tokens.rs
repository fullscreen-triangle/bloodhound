//! Token lifecycle: every account's token tracked, checked, and replaced before it
//! expires wherever the forge allows it.
//!
//! Tokens live only in KeePassXC. tracker never opens the vault: it reaches a token
//! through the paired `git-credential-keepassxc` helper (KeePassXC decides whether
//! to answer), holds it in memory for the API call that needs it, and writes a
//! replacement back the same way. On disk it keeps metadata only —
//! `~/.tracker/tokens.json`: token id, scopes, expiry, last check and rotation.
//!
//! What each forge allows:
//! * **GitLab** rotates a token through its own API (`personal_access_tokens/self/rotate`),
//!   revoking the old one in the same step, so tracker renews it unattended.
//! * **GitHub** and **Gitea** have no API to issue or rotate a token. Their tokens
//!   can be made non-expiring; tracker checks they still work, reads GitHub's
//!   expiry date when there is one, and says ahead of time when to reissue by hand.

use crate::error::{Result, TrackerError};
use crate::profile::{Account, Forge, Profile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Renew (or warn) when a token has this many days or fewer left.
pub const RENEW_WITHIN_DAYS: i64 = 14;
/// Lifetime given to a rotated GitLab token.
pub const LIFETIME_DAYS: i64 = 90;

const HELPER: &str = "git-credential-keepassxc";

// ── dates, without a date crate ─────────────────────────────────────────────

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 }, m, d)
}

fn today() -> i64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    secs.div_euclid(86_400)
}

fn format_day(z: i64) -> String {
    let (y, m, d) = civil_from_days(z);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The day of a `YYYY-MM-DD…` string (anything after the date is ignored).
fn parse_day(s: &str) -> Option<i64> {
    let s = s.get(..10)?;
    let mut it = s.split('-').map(|p| p.parse::<i64>().ok());
    Some(days_from_civil(it.next()??, it.next()??, it.next()??))
}

// ── the helper: the only way to a token ─────────────────────────────────────

fn helper(args: &[&str], input: &str) -> Result<(bool, String, String)> {
    let mut child = Command::new(HELPER)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TrackerError::Keeper(format!("could not run {HELPER}: {e}; `cargo install git-credential-keepassxc`")))?;
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(input.as_bytes())?;
    let out = child.wait_with_output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn unlock_args(wait: bool) -> Vec<&'static str> {
    // Waiting asks KeePassXC to show its unlock dialog and retries for about a minute.
    if wait {
        vec!["--unlock", "20,3000"]
    } else {
        Vec::new()
    }
}

fn helper_failure(stderr: &str) -> TrackerError {
    if stderr.contains("DatabaseNotOpened") || stderr.to_lowercase().contains("locked") {
        TrackerError::KeeperLocked
    } else {
        TrackerError::Keeper(stderr.trim().lines().last().unwrap_or("").to_string())
    }
}

/// The token KeePassXC holds for this account, if any.
fn keeper_get(a: &Account, wait: bool) -> Result<Option<String>> {
    let mut args = unlock_args(wait);
    args.push("get");
    let input = format!("protocol=https\nhost={}\nusername={}\n\n", a.host, a.user);
    let (ok, stdout, stderr) = helper(&args, &input)?;
    if !ok {
        return if stderr.contains("NoLoginsFound") {
            Ok(None)
        } else {
            Err(helper_failure(&stderr))
        };
    }
    Ok(stdout
        .lines()
        .find_map(|l| l.strip_prefix("password="))
        .filter(|p| !p.is_empty())
        .map(str::to_string))
}

/// Write `token` into KeePassXC for this account, updating its entry in place.
fn keeper_store(a: &Account, token: &str, wait: bool) -> Result<()> {
    let mut args = unlock_args(wait);
    args.push("store");
    let input = format!(
        "protocol=https\nhost={}\nusername={}\npassword={token}\n\n",
        a.host, a.user
    );
    let (ok, _, stderr) = helper(&args, &input)?;
    // The helper exits 0 even when KeePassXC refused, so read what it said.
    if !ok || stderr.contains("ERRO") {
        return Err(helper_failure(&stderr));
    }
    Ok(())
}

// ── HTTP through curl; the token reaches it on stdin only ────────────────────

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Response {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

fn http(method: &str, url: &str, auth: &str, form: &[(&str, String)]) -> Result<Response> {
    let mut args: Vec<String> = ["-sS", "-i", "-m", "30", "-X", method, "-H", "@-"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (k, v) in form {
        args.push("--data-urlencode".into());
        args.push(format!("{k}={v}"));
    }
    args.push(url.to_string());
    let mut child = Command::new("curl")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TrackerError::Forge(format!("could not run curl: {e}")))?;
    let headers = format!("Accept: application/json\nUser-Agent: tracker\n{auth}\n");
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(headers.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(TrackerError::Forge(format!(
            "{url}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let mut rest = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    let mut block = String::new();
    // Skip interim responses (100 Continue, redirects): the last header block is the answer.
    while rest.starts_with("HTTP/") {
        let (head, tail) = rest.split_once("\n\n").unwrap_or((rest.as_str(), ""));
        block = head.to_string();
        rest = tail.to_string();
    }
    let mut lines = block.lines();
    let status = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Ok(Response { status, headers, body: rest })
}

fn auth_header(kind: Forge, token: &str) -> String {
    match kind {
        Forge::Gitlab => format!("PRIVATE-TOKEN: {token}"),
        Forge::Github => format!("Authorization: Bearer {token}"),
        Forge::Gitea => format!("Authorization: token {token}"),
    }
}

fn api_base(a: &Account) -> String {
    match a.kind {
        Forge::Github if a.host == "github.com" => "https://api.github.com".into(),
        Forge::Github => format!("https://{}/api/v3", a.host),
        Forge::Gitlab => format!("https://{}/api/v4", a.host),
        Forge::Gitea => format!("https://{}/api/v1", a.host),
    }
}

/// The page where a new token for this account is made, pre-filled where the forge allows.
pub fn issue_url(a: &Account) -> String {
    match a.kind {
        Forge::Github => format!(
            "https://{}/settings/tokens/new?description=tracker&scopes=repo,read:org",
            a.host
        ),
        Forge::Gitlab => format!(
            "https://{}/-/user_settings/personal_access_tokens?name=tracker&scopes=api,read_repository,write_repository",
            a.host
        ),
        Forge::Gitea => format!("https://{}/user/settings/applications", a.host),
    }
}

// ── metadata ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Record {
    pub token_id: Option<u64>,
    pub scopes: Vec<String>,
    /// `YYYY-MM-DD`, or none for a token that does not expire (or whose expiry is unknown).
    pub expires_at: Option<String>,
    pub checked_at: Option<String>,
    pub rotated_at: Option<String>,
}

fn records_path() -> PathBuf {
    Profile::path().with_file_name("tokens.json")
}

fn load_records() -> BTreeMap<String, Record> {
    std::fs::read_to_string(records_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_records(r: &BTreeMap<String, Record>) -> Result<()> {
    let path = records_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(r).expect("serialisable"))?;
    Ok(())
}

// ── the lifecycle ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Works, and is not close to expiring.
    Ok,
    /// Replaced just now; the new token is in KeePassXC.
    Rotated,
    /// Works, but expires soon and this forge cannot rotate it: reissue by hand.
    Expiring,
    /// KeePassXC holds nothing for this account.
    Missing,
    /// The forge rejects the token: expired or revoked.
    Invalid,
    /// Could not be checked this time (KeePassXC locked, network down, …).
    Unchecked,
}

#[derive(Debug, Serialize)]
pub struct Status {
    pub account: String,
    pub host: String,
    pub state: State,
    pub expires_at: Option<String>,
    pub days_left: Option<i64>,
    pub scopes: Vec<String>,
    /// What, if anything, the user needs to do.
    pub action: Option<String>,
    /// Where a new token is made, when one is needed.
    pub issue_url: Option<String>,
}

pub struct Options {
    /// Rotate tokens that are due (otherwise only check them).
    pub rotate: bool,
    /// Rotate GitLab tokens even when not due.
    pub force: bool,
    /// Wait for KeePassXC to be unlocked (interactive use) rather than skip.
    pub wait_unlock: bool,
}

fn status(a: &Account, state: State, rec: &Record, action: Option<String>) -> Status {
    let days_left = rec.expires_at.as_deref().and_then(parse_day).map(|d| d - today());
    let needs_new = matches!(state, State::Missing | State::Invalid | State::Expiring);
    Status {
        account: a.id.clone(),
        host: a.host.clone(),
        state,
        expires_at: rec.expires_at.clone(),
        days_left,
        scopes: rec.scopes.clone(),
        action,
        issue_url: needs_new.then(|| issue_url(a)),
    }
}

fn due(rec: &Record) -> bool {
    rec.expires_at
        .as_deref()
        .and_then(parse_day)
        .is_some_and(|d| d - today() <= RENEW_WITHIN_DAYS)
}

fn manual(a: &Account) -> String {
    format!(
        "make a new token at the page below and store it with `tracker tokens set {}`",
        a.id
    )
}

fn check_one(a: &Account, rec: &mut Record, o: &Options) -> Result<Status> {
    let token = match keeper_get(a, o.wait_unlock)? {
        Some(t) => t,
        None => return Ok(status(a, State::Missing, rec, Some(manual(a)))),
    };
    let auth = auth_header(a.kind, &token);
    rec.checked_at = Some(format_day(today()));
    match a.kind {
        Forge::Gitlab => {
            let r = http("GET", &format!("{}/personal_access_tokens/self", api_base(a)), &auth, &[])?;
            if r.status == 401 {
                return Ok(status(a, State::Invalid, rec, Some(manual(a))));
            }
            let v = r.json();
            rec.token_id = v["id"].as_u64();
            rec.expires_at = v["expires_at"].as_str().map(str::to_string);
            rec.scopes = scopes(&v["scopes"]);
            if o.rotate && (o.force || due(rec)) {
                return rotate_gitlab(a, rec, &auth, o);
            }
            Ok(status(a, State::Ok, rec, None))
        }
        Forge::Github => {
            let r = http("GET", &format!("{}/user", api_base(a)), &auth, &[])?;
            if r.status == 401 {
                return Ok(status(a, State::Invalid, rec, Some(manual(a))));
            }
            rec.expires_at = r
                .header("github-authentication-token-expiration")
                .and_then(parse_day)
                .map(format_day);
            rec.scopes = r
                .header("x-oauth-scopes")
                .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default();
            if due(rec) {
                let msg = format!("GitHub cannot rotate tokens by API; {}", manual(a));
                return Ok(status(a, State::Expiring, rec, Some(msg)));
            }
            Ok(status(a, State::Ok, rec, None))
        }
        Forge::Gitea => {
            let r = http("GET", &format!("{}/user", api_base(a)), &auth, &[])?;
            if r.status == 401 {
                return Ok(status(a, State::Invalid, rec, Some(manual(a))));
            }
            rec.expires_at = None;
            Ok(status(a, State::Ok, rec, None))
        }
    }
}

fn scopes(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn rotate_gitlab(a: &Account, rec: &mut Record, auth: &str, o: &Options) -> Result<Status> {
    // Prove KeePassXC will take a write *before* GitLab revokes the current token.
    let current = keeper_get(a, o.wait_unlock)?
        .ok_or_else(|| TrackerError::Keeper("the token vanished from KeePassXC mid-rotation".into()))?;
    keeper_store(a, &current, o.wait_unlock)?;

    let expires = format_day(today() + LIFETIME_DAYS);
    let form = [("expires_at", expires.clone())];
    let base = api_base(a);
    let mut r = http("POST", &format!("{base}/personal_access_tokens/self/rotate"), auth, &form)?;
    if r.status == 404 {
        // GitLab before 16.10 can only rotate by id.
        let id = rec.token_id.ok_or_else(|| TrackerError::Forge("token id unknown; cannot rotate".into()))?;
        r = http("POST", &format!("{base}/personal_access_tokens/{id}/rotate"), auth, &form)?;
    }
    if r.status == 403 {
        let msg = "the token lacks the `api` (or `self_rotate`) scope, so GitLab will not let it rotate itself; \
                   make one with that scope"
            .to_string();
        return Ok(status(a, State::Ok, rec, Some(msg)));
    }
    let v = r.json();
    let new = match (r.status, v["token"].as_str()) {
        (200 | 201, Some(t)) => t.to_string(),
        _ => {
            return Err(TrackerError::Forge(format!(
                "{} refused the rotation ({}): {}",
                a.host,
                r.status,
                r.body.chars().take(200).collect::<String>()
            )))
        }
    };
    // GitLab has already revoked the old token: the new one must not be lost.
    if let Err(e) = keeper_store(a, &new, true) {
        let rescue = records_path().with_file_name(format!("rescue-{}.token", a.id));
        std::fs::write(&rescue, &new)?;
        return Err(TrackerError::Keeper(format!(
            "rotated {}, but KeePassXC would not take the new token ({e}). It is saved in {} — \
             put it into KeePassXC by hand, then delete that file",
            a.id,
            rescue.display()
        )));
    }
    rec.token_id = v["id"].as_u64();
    rec.expires_at = v["expires_at"].as_str().map(str::to_string).or(Some(expires));
    rec.scopes = scopes(&v["scopes"]);
    rec.rotated_at = Some(format_day(today()));
    Ok(status(a, State::Rotated, rec, None))
}

/// Check every account's token, rotating the ones that are due when `o.rotate`.
pub fn refresh(only: &[String], o: &Options) -> Result<Vec<Status>> {
    let profile = Profile::require()?;
    let mut records = load_records();
    let mut out = Vec::new();
    for a in &profile.accounts {
        if !only.is_empty() && !only.contains(&a.id) {
            continue;
        }
        let rec = records.entry(a.id.clone()).or_default();
        let s = match check_one(a, rec, o) {
            Ok(s) => s,
            // A rotation that lost its token is never swallowed.
            Err(e @ TrackerError::Keeper(_)) if e.to_string().contains("rescue") => {
                save_records(&records)?;
                return Err(e);
            }
            Err(e) => {
                let mut s = status(a, State::Unchecked, rec, None);
                s.action = Some(e.to_string());
                s
            }
        };
        out.push(s);
    }
    save_records(&records)?;
    Ok(out)
}

/// Store a token the user just made, after checking the forge accepts it.
pub fn set(account: &str, token: &str) -> Result<Status> {
    let profile = Profile::require()?;
    let a = profile
        .account(account)
        .ok_or_else(|| TrackerError::Profile(format!("no account {account:?} in the profile")))?;
    let token = token.trim();
    let probe = match a.kind {
        Forge::Gitlab => format!("{}/personal_access_tokens/self", api_base(a)),
        Forge::Github | Forge::Gitea => format!("{}/user", api_base(a)),
    };
    let r = http("GET", &probe, &auth_header(a.kind, token), &[])?;
    if r.status != 200 {
        return Err(TrackerError::Forge(format!(
            "{} does not accept that token (HTTP {}); nothing was stored",
            a.host, r.status
        )));
    }
    keeper_store(a, token, true)?;
    let o = Options {
        rotate: false,
        force: false,
        wait_unlock: true,
    };
    Ok(refresh(&[account.to_string()], &o)?.remove(0))
}

// ── scheduling ──────────────────────────────────────────────────────────────

pub const TASK_NAME: &str = "tracker token refresh";

/// Register (or remove) a daily check that renews tokens before they expire.
pub fn schedule(on: bool) -> Result<String> {
    if !cfg!(windows) {
        return Ok("add to your crontab:  17 9 * * *  tracker tokens refresh --unattended".into());
    }
    let exe = std::env::current_exe()?;
    let args: Vec<String> = if on {
        vec![
            "/Create".into(), "/F".into(), "/SC".into(), "DAILY".into(), "/ST".into(), "09:17".into(),
            "/TN".into(), TASK_NAME.into(),
            "/TR".into(), format!("\"{}\" tokens refresh --unattended", exe.display()),
        ]
    } else {
        vec!["/Delete".into(), "/F".into(), "/TN".into(), TASK_NAME.into()]
    };
    let out = Command::new("schtasks").args(&args).output()?;
    if !out.status.success() {
        return Err(TrackerError::Keeper(format!(
            "schtasks failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(if on {
        format!("Scheduled daily at 09:17: {} tokens refresh --unattended", exe.display())
    } else {
        "Removed the daily token refresh.".into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip_and_count_days() {
        for z in [-1000, 0, 19_000, 20_725, 30_000] {
            assert_eq!(parse_day(&format_day(z)), Some(z));
        }
        assert_eq!(format_day(0), "1970-01-01");
        assert_eq!(parse_day("2026-09-29 10:11:12 UTC"), parse_day("2026-09-29"));
        assert_eq!(parse_day("2024-03-01").unwrap() - parse_day("2024-02-28").unwrap(), 2);
        assert_eq!(parse_day("garbage"), None);
    }

    #[test]
    fn due_means_within_the_renewal_window() {
        let at = |d: i64| Record {
            expires_at: Some(format_day(today() + d)),
            ..Default::default()
        };
        assert!(due(&at(RENEW_WITHIN_DAYS)));
        assert!(due(&at(-1)));
        assert!(!due(&at(RENEW_WITHIN_DAYS + 1)));
        assert!(!due(&Record::default()), "a non-expiring token is never due");
    }
}
