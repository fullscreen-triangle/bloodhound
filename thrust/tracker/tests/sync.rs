//! End-to-end tests of `tracker sync` against real git repositories: a canonical
//! repo, two bare "origins" standing in for the university and company GitLabs, and
//! collaborator clones of them.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct World {
    _dir: tempfile::TempDir,
    root: PathBuf,
    config: PathBuf,
    canon: PathBuf,
    uni: PathBuf,
    co: PathBuf,
}

fn slash(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

impl World {
    fn env(&self, cmd: &mut Command, who: &str) {
        cmd.env("GIT_CONFIG_GLOBAL", &self.config)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("TRACKER_PROFILE", self.root.join("profile.toml"))
            .env("GIT_AUTHOR_NAME", who)
            .env("GIT_AUTHOR_EMAIL", format!("{}@example.org", who.to_lowercase()))
            .env("GIT_COMMITTER_NAME", who)
            .env("GIT_COMMITTER_EMAIL", format!("{}@example.org", who.to_lowercase()));
    }

    fn git_as(&self, who: &str, dir: &Path, args: &[&str]) -> String {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(dir).args(args);
        self.env(&mut cmd, who);
        let out = cmd.output().expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git(&self, dir: &Path, args: &[&str]) -> String {
        self.git_as("Me", dir, args)
    }

    fn try_git(&self, dir: &Path, args: &[&str]) -> Output {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(dir).args(args);
        self.env(&mut cmd, "Me");
        cmd.output().expect("git runs")
    }

    fn tracker(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tracker"));
        cmd.current_dir(&self.canon).arg("sync").args(args);
        self.env(&mut cmd, "Me");
        cmd.output().expect("tracker runs")
    }

    fn sync_ok(&self, args: &[&str]) -> String {
        let out = self.tracker(args);
        assert!(
            out.status.success(),
            "tracker sync {args:?} failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn sync_err(&self, args: &[&str]) -> String {
        let out = self.tracker(args);
        assert!(!out.status.success(), "tracker sync {args:?} unexpectedly succeeded");
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    fn write(&self, dir: &Path, rel: &str, content: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    /// Commit with an explicit name and email, as a forge's web editor would.
    fn commit_all_ident(&self, name: &str, email: &str, dir: &Path, message: &str) {
        self.git(dir, &["add", "-A"]);
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(dir).args(["commit", "-q", "-m", message]);
        self.env(&mut cmd, name);
        cmd.env("GIT_AUTHOR_EMAIL", email).env("GIT_COMMITTER_EMAIL", email);
        assert!(cmd.output().unwrap().status.success());
    }

    fn commit_all(&self, who: &str, dir: &Path, message: &str) -> String {
        self.git_as(who, dir, &["add", "-A"]);
        self.git_as(who, dir, &["commit", "-q", "-m", message]);
        self.git(dir, &["rev-parse", "HEAD"])
    }

    fn files(&self, repo: &Path, rev: &str) -> Vec<String> {
        let out = self.git(repo, &["ls-tree", "-r", "--name-only", rev]);
        out.lines().map(str::to_string).collect()
    }

    fn tip(&self, repo: &Path) -> String {
        self.git(repo, &["rev-parse", "main"])
    }

    /// A collaborator's clone of a remote.
    fn clone(&self, remote: &Path, name: &str) -> PathBuf {
        let dir = self.root.join(name);
        self.git(&self.root, &["clone", "-q", &slash(remote), &slash(&dir)]);
        dir
    }
}

/// Canonical repo with public, company-only and private paths, synced once.
fn world() -> World {
    world_with(None)
}

/// As `world`, optionally with a profile giving the uni origin its own identity.
fn world_with(profile: Option<&str>) -> World {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let config = root.join("gitconfig");
    std::fs::write(&config, "").unwrap();
    let w = World {
        canon: root.join("canon"),
        uni: root.join("uni.git"),
        co: root.join("co.git"),
        _dir: dir,
        root,
        config,
    };
    w.git(&w.root, &["init", "-q", "--bare", "-b", "main", &slash(&w.uni)]);
    w.git(&w.root, &["init", "-q", "--bare", "-b", "main", &slash(&w.co)]);
    w.git(&w.root, &["init", "-q", "-b", "main", &slash(&w.canon)]);

    if let Some(p) = profile {
        std::fs::write(w.root.join("profile.toml"), p).unwrap();
    }
    // Local paths carry no host, so the uni origin names its account explicitly.
    let uni_account = if profile.is_some() { "account = \"uni\"\n" } else { "" };
    let manifest = format!(
        "branch = \"main\"\n\
         [remotes.uni]\nurl = \"{}\"\nsees = [\"uni\"]\n{uni_account}\
         [remotes.co]\nurl = \"{}\"\nsees = [\"uni\", \"co\"]\n\
         [hide]\n\"co/\" = \"co\"\n\"notes/private/\" = \"private\"\n",
        slash(&w.uni),
        slash(&w.co)
    );
    w.write(&w.canon, ".sync.toml", &manifest);
    w.write(&w.canon, "paper.md", "line one\nline two\nline three\n");
    w.commit_all("Me", &w.canon, "Start the paper");
    w.write(&w.canon, "co/pricing.md", "secret pricing\n");
    w.commit_all("Me", &w.canon, "Add company pricing");
    w.write(&w.canon, "notes/private/diary.md", "private\n");
    w.commit_all("Me", &w.canon, "Add private notes");
    w.sync_ok(&["run"]);
    w
}

#[test]
fn each_remote_sees_only_what_it_may() {
    let w = world();
    assert_eq!(w.files(&w.uni, "main"), vec!["paper.md"]);
    assert_eq!(w.files(&w.co, "main"), vec!["co/pricing.md", "paper.md"]);
    // The hidden-only commit never reaches uni at all.
    let uni_log = w.git(&w.uni, &["log", "--format=%s", "main"]);
    assert_eq!(uni_log, "Start the paper");
    // The manifest, which names the hidden paths, reaches nobody.
    assert!(!w.files(&w.co, "main").contains(&".sync.toml".to_string()));
}

#[test]
fn syncing_again_changes_nothing() {
    let w = world();
    let (uni, co) = (w.tip(&w.uni), w.tip(&w.co));
    let out = w.sync_ok(&["run"]);
    assert!(out.contains("up to date"), "{out}");
    assert_eq!((w.tip(&w.uni), w.tip(&w.co)), (uni, co));
}

#[test]
fn projection_is_rebuilt_without_its_cache() {
    let w = world();
    let uni = w.tip(&w.uni);
    std::fs::remove_dir_all(w.canon.join(".git").join("tracker-sync")).unwrap();
    let out = w.sync_ok(&["run"]);
    assert!(out.contains("up to date"), "{out}");
    assert_eq!(w.tip(&w.uni), uni);
}

#[test]
fn hidden_only_work_goes_only_where_it_may() {
    let w = world();
    let uni = w.tip(&w.uni);
    w.write(&w.canon, "co/pricing.md", "new pricing\n");
    w.commit_all("Me", &w.canon, "Reprice");
    w.sync_ok(&["run"]);
    assert_eq!(w.tip(&w.uni), uni);
    let co_msg = w.git(&w.co, &["log", "-1", "--format=%s", "main"]);
    assert_eq!(co_msg, "Reprice");
}

#[test]
fn collaborator_commits_come_back_and_flow_on() {
    let w = world();
    let collab = w.clone(&w.uni, "supervisor");
    w.write(&collab, "paper.md", "line one\nline two, revised\nline three\n");
    let theirs = w.commit_all("Supervisor", &collab, "Revise line two");
    w.git(&collab, &["push", "-q", "origin", "main"]);

    w.sync_ok(&["run"]);

    // Canonical has their change, keeps its hidden files, and credits them.
    let paper = std::fs::read_to_string(w.canon.join("paper.md")).unwrap();
    assert!(paper.contains("revised"));
    assert!(w.canon.join("co/pricing.md").exists());
    assert_eq!(w.git(&w.canon, &["log", "-1", "--format=%an", "main"]), "Supervisor");
    let body = w.git(&w.canon, &["log", "-1", "--format=%B", "main"]);
    assert!(body.contains(&format!("Sync-Origin: uni {theirs}")), "{body}");

    // Nothing moved meanwhile, so their own commit is uni's tip — no extra commit.
    assert_eq!(w.tip(&w.uni), theirs);
    // It flowed on to the company remote, credited, without the sync trailer.
    assert_eq!(w.git(&w.co, &["log", "-1", "--format=%an", "main"]), "Supervisor");
    let co_body = w.git(&w.co, &["log", "-1", "--format=%B", "main"]);
    assert!(!co_body.contains("Sync-Origin"), "{co_body}");
    assert!(w.git(&w.co, &["show", "main:paper.md"]).contains("revised"));
}

#[test]
fn concurrent_work_on_both_sides_merges() {
    let w = world();
    let collab = w.clone(&w.uni, "supervisor");
    w.write(&collab, "paper.md", "line one\nline two, revised\nline three\n");
    let theirs = w.commit_all("Supervisor", &collab, "Revise line two");
    w.git(&collab, &["push", "-q", "origin", "main"]);

    w.write(&w.canon, "results.md", "results\n");
    w.commit_all("Me", &w.canon, "Add results");

    w.sync_ok(&["run"]);

    let uni_tip = w.tip(&w.uni);
    assert_ne!(uni_tip, theirs);
    // Their commit is kept in uni's history; nothing was force-pushed.
    assert!(w.try_git(&w.uni, &["merge-base", "--is-ancestor", &theirs, &uni_tip]).status.success());
    assert_eq!(w.files(&w.uni, "main"), vec!["paper.md", "results.md"]);
    assert!(w.git(&w.uni, &["show", "main:paper.md"]).contains("revised"));
}

#[test]
fn a_collaborator_cannot_write_into_a_hidden_path() {
    let w = world();
    let collab = w.clone(&w.uni, "supervisor");
    w.write(&collab, "co/idea.md", "an idea\n");
    w.commit_all("Supervisor", &collab, "Add idea");
    w.git(&collab, &["push", "-q", "origin", "main"]);
    let before = w.tip(&w.canon);
    let err = w.sync_err(&["run"]);
    assert!(err.contains("hides from uni"), "{err}");
    assert_eq!(w.tip(&w.canon), before);
}

#[test]
fn conflicts_stop_and_resolve_by_hand() {
    let w = world();
    let collab = w.clone(&w.uni, "supervisor");
    w.write(&collab, "paper.md", "line one, theirs\nline two\nline three\n");
    let theirs = w.commit_all("Supervisor", &collab, "Their first line");
    w.git(&collab, &["push", "-q", "origin", "main"]);
    w.write(&w.canon, "paper.md", "line one, mine\nline two\nline three\n");
    let mine = w.commit_all("Me", &w.canon, "My first line");

    let err = w.sync_err(&["run"]);
    assert!(err.contains("conflicts") && err.contains("paper.md"), "{err}");
    assert_eq!(w.tip(&w.canon), mine);

    let out = w.sync_ok(&["resolve", "uni"]);
    assert!(out.contains("paper.md"), "{out}");
    w.write(&w.canon, "paper.md", "line one, both\nline two\nline three\n");
    w.git(&w.canon, &["add", "paper.md"]);
    w.sync_ok(&["resolve", "--continue"]);
    assert!(w.canon.join("co/pricing.md").exists());

    w.sync_ok(&["run"]);
    let uni_tip = w.tip(&w.uni);
    assert!(w.try_git(&w.uni, &["merge-base", "--is-ancestor", &theirs, &uni_tip]).status.success());
    assert!(w.git(&w.uni, &["show", "main:paper.md"]).contains("line one, both"));
}

#[test]
fn mixed_commits_need_a_message_fit_for_each_remote() {
    let w = world();
    w.write(&w.canon, "paper.md", "line one\nline two\nline three\nline four\n");
    w.write(&w.canon, "co/pricing.md", "cheaper\n");
    let c = w.commit_all("Me", &w.canon, "Cut the price and extend the paper");

    let err = w.sync_err(&["run"]);
    assert!(err.contains("uni: nothing pushed"), "{err}");
    assert!(err.contains("Cut the price"), "{err}");
    assert_eq!(w.git(&w.uni, &["log", "-1", "--format=%s", "main"]), "Start the paper");

    w.sync_ok(&["message", &c, "--remote", "uni", "Extend the paper"]);
    w.sync_ok(&["run"]);
    assert_eq!(w.git(&w.uni, &["log", "-1", "--format=%s", "main"]), "Extend the paper");
    assert_eq!(
        w.git(&w.co, &["log", "-1", "--format=%s", "main"]),
        "Cut the price and extend the paper"
    );
}

#[test]
fn direct_pushes_of_the_canonical_branch_are_refused() {
    let w = world();
    let out = w.try_git(&w.canon, &["push", &slash(&w.uni), "main"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("tracker"));
    assert_eq!(w.files(&w.uni, "main"), vec!["paper.md"]);
}

#[test]
fn status_changes_nothing() {
    let w = world();
    w.write(&w.canon, "results.md", "results\n");
    w.commit_all("Me", &w.canon, "Add results");
    let uni = w.tip(&w.uni);
    let out = w.sync_ok(&["status"]);
    assert!(out.contains("would push"), "{out}");
    assert_eq!(w.tip(&w.uni), uni);
}

const PROFILE: &str = r#"
[identity]
name = "Me"
email = "me@example.org"

[accounts.uni]
kind = "gitea"
host = "uni.test"
user = "me"
name = "Me at Uni"
email = "me@uni.test"
"#;

fn who(w: &World, repo: &Path) -> String {
    w.git(repo, &["log", "-1", "--format=%an <%ae> / %cn <%ce>", "main"])
}

#[test]
fn each_origin_sees_your_commits_under_its_own_identity() {
    let w = world_with(Some(PROFILE));
    assert_eq!(who(&w, &w.uni), "Me at Uni <me@uni.test> / Me at Uni <me@uni.test>");
    // The company origin has no account in this profile, so it sees you as you are.
    assert_eq!(who(&w, &w.co), "Me <me@example.org> / Me <me@example.org>");

    // Rebuilt without its cache, the per-host projection is still recognised as ours.
    let uni = w.tip(&w.uni);
    std::fs::remove_dir_all(w.canon.join(".git").join("tracker-sync")).unwrap();
    let out = w.sync_ok(&["run"]);
    assert!(out.contains("re-recognised") && out.contains("up to date"), "{out}");
    assert_eq!(w.tip(&w.uni), uni);

    // A collaborator keeps their own identity everywhere.
    let collab = w.clone(&w.uni, "supervisor");
    w.write(&collab, "paper.md", "line one\nline two, revised\nline three\n");
    w.commit_all("Supervisor", &collab, "Revise line two");
    w.git(&collab, &["push", "-q", "origin", "main"]);
    w.sync_ok(&["run"]);
    assert!(who(&w, &w.uni).starts_with("Supervisor <"));
    assert!(who(&w, &w.co).starts_with("Supervisor <"));

    // Your own edit made on the uni host comes home as you, and reaches the
    // company origin as you — never under the uni identity.
    let web = w.clone(&w.uni, "web");
    w.write(&web, "paper.md", "line one\nline two, revised\nline three\nline four\n");
    w.commit_all_ident("Me at Uni", "me@uni.test", &web, "Add line four");
    w.git(&web, &["push", "-q", "origin", "main"]);
    let edit = w.tip(&w.uni);
    w.sync_ok(&["run"]);
    assert_eq!(who(&w, &w.canon), "Me <me@example.org> / Me <me@example.org>");
    assert_eq!(who(&w, &w.co), "Me <me@example.org> / Me <me@example.org>");
    assert_eq!(w.tip(&w.uni), edit);
}
