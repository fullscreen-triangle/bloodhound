# tracker — repo-federation tracker

A Rust CLI that tracks a **group of repositories and their conserved sense/goal** —
not just commits, but *what each repo is fundamentally about* and whether that has
moved. Built for a research group with many repos.

It is one of three composable, per-project-installable tools; it reimplements
neither of the other two:

| Organ | Tool | Role |
|---|---|---|
| **Search** | [`purpose`](../../../semantics/purpose) | Answers "what/where/sense" by *searching, not fetching*. |
| **Execution** | network-yield CLI *(in development)* | Runs specific repo code in the cloud (Codespaces / cloud compute). |
| **Tracking** | **`tracker`** (this tool) | Federates repos, holds each repo's invariant χ, coordinates the agents. |

Design doc: [`../docs/tracker/repo-federation-tracker-design.md`](../docs/tracker/repo-federation-tracker-design.md).

## What "sense/goal" means: χ, the character invariant

A repo is a finite weighted graph (its files/sections, joined by containment and
cross-reference). Its **character invariant χ** is the *minimum cut-residual* — the
least-cost way to split it into unrelated pieces, in the currency of its own
structure. χ is:

- **positive** (a repo with structure has a non-collapsible sense),
- **conserved** under relabelling (renaming files/symbols doesn't change it), and
- **non-local** (it names a *region* that most cheaply severs, never one file).

χ is computed over the repo's **largest connected component** (its principal body of
work) and the number of disconnected **fragments** is reported alongside — a repo's
raw index is naturally split between, e.g., docs and code, and that split is a fact
worth surfacing, not hiding.

χ is *cached only as a change-detector*. Every actual answer is produced by a fresh
`purpose ask` against the current index — the tool never serves a stored summary
(the "search-not-fetch" discipline).

## Install

```bash
cd thrust/tracker
cargo install --path .
```

Requires the `purpose` CLI on PATH (`purpose --version`). Execution (`run`) also
requires the network-yield CLI once its interface is finalised; until then the tool
is a full read-only tracker/search federation and `run` degrades gracefully.

## Use

```bash
tracker init                       # create the federation here (.tracker/)
tracker add ../my-repo             # register a repo; indexes it via `purpose`
tracker add ../other --name lib    # …with an explicit name
tracker list                       # repos, current χ, committed count m

tracker sense my-repo              # the repo's standing sense/goal (χ + salient files)
tracker sense my-repo "where is the parser"   # fresh search within one repo
tracker ask "S-entropy coordinate"            # search across the whole federation

tracker drift my-repo              # recompute χ; report whether the sense has moved
tracker run my-repo "run tests"    # hand a goal to the execution organ (when wired)
```

## Using it from anywhere: scripts, other programs, AI agents

Everything the tool can answer or do is one registry of named operations. Each is
a **question** (reads) or an **action** (changes a repo, the federation or an
origin), takes a JSON object and returns one. There are three ways in, all
reaching the same code:

| Caller | Entry point |
|---|---|
| a script, CI job, or any program | `tracker describe`, `tracker call <op> '<json>'` |
| an AI agent (Claude Code, Claude Desktop, any MCP client) | `tracker mcp` |
| a Rust crate | `tracker::api::call(op, json)` / `tracker::api::ops()` |

**Install** once per machine; it then works from any directory:

```bash
cargo install --git https://github.com/fullscreen-triangle/bloodhound tracker
# or, from a checkout:  cargo install --path thrust/tracker
```

**Programs.** `tracker describe` lists every operation with its kind, whether it
touches the network, and its JSON Schema. `tracker call` runs one and prints JSON.
A failure prints `{"error": {"code": "...", "message": "..."}}` and exits
non-zero. Codes are stable, so callers can branch on them.

```bash
tracker call sync_visibility '{"paths": ["bitspark/pricing.md"]}'
tracker call sync_hide '{"pattern": "bitspark/", "label": "bitspark"}'
tracker call sync_status '{"repo": "bloodhound"}'
echo '{"question": "entropy"}' | tracker call federation_ask -
```

**AI agents.** `tracker mcp` serves every operation as an MCP tool over stdio.
Questions are marked read-only. Actions that reach origins are marked open-world,
so the client can ask before running them. To give an agent working in some other
project the tool, add to that project's `.mcp.json`:

```json
{ "mcpServers": { "tracker": { "command": "tracker", "args": ["mcp"] } } }
```

The server treats the directory it starts in as the current repo (override with
`--root <dir>`). Any operation also accepts `"repo"`: a path, or the name of a repo
tracked in the federation. A federation found by walking up from the current
directory is used first; otherwise `~/.tracker/federation.json`.

| Operation | Kind | Does |
|---|---|---|
| `federation_list` | question | tracked repos, χ, committed counts |
| `repo_sense` | question | what a repo is about, or a `purpose` search of it |
| `federation_ask` | question | one search across every tracked repo |
| `repo_drift` | action | recompute χ, record it, say whether the sense moved |
| `profile_show` | question | your accounts and the identity each host sees |
| `profile_repos` | question | every repo on every account, copies grouped |
| `sync_manifest` | question | a repo's origins and hide rules |
| `sync_visibility` | question | which origins may see given paths |
| `sync_status` | question | what a sync would pull and push (changes nothing) |
| `sync_run` | action | pull collaborators' commits, push each origin its view |
| `sync_message` | action | the message an origin sees for a mixed commit |
| `sync_resolve` | action | hand-resolve a blocked collaborator commit |
| `sync_hide` / `sync_unhide` | action | change what is hidden; commits `.sync.toml` |
| `sync_set_remote` | action | add or change an origin |

The error codes that ask the caller to act: `sync_conflict` and
`hidden_from_remote` → `sync_resolve`; `message_leak` → `sync_message`;
`purpose_missing` → install `purpose`; `no_federation` → `tracker init` somewhere
(e.g. in your home directory).

## You, across every forge: `tracker profile`

One profile for all your accounts, e.g. GitHub, gitlab.com, a company GitLab and a
university Gitea. It lives outside every repo (`~/.tracker/profile.toml`, or
`$TRACKER_PROFILE`) and holds **no secrets**. Credentials come from KeePassXC
through git's own credential protocol.

```toml
[identity]                      # you, on your canonical branches
name  = "Your Name"
email = "you@example.org"
also  = ["old@example.org"]     # further addresses that are also you

[accounts.work]
kind  = "gitlab"                # github | gitlab | gitea
host  = "gitlab.example-company.com"
user  = "you"
name  = "Your Name"             # how this host sees your commits
email = "you@example-company.com" # (default: [identity])
```

```bash
tracker profile init            # write a template
tracker profile show            # accounts, and who each host sees you as
tracker profile repos           # every repo on every account; copies grouped
tracker profile repos --json
tracker profile git-setup       # show the git config that hands credentials to KeePassXC
tracker profile git-setup --apply
```

**Per-host identity.** `tracker sync` gives your own commits the identity of the
account an origin belongs to: the Bitspark origin sees your Bitspark address, the
university sees your university address. Collaborators' commits keep their own
identities. Your edits made directly on a host come home under your canonical
identity. A manifest remote is matched to an account by host, or explicitly with
`account = "<id>"`.

**KeePassXC.** `git-setup --apply` does two things for each host. It replaces the
inherited credential helper (e.g. Git Credential Manager) with `keepassxc`, which is
[`git-credential-keepassxc`](https://github.com/Frederick888/git-credential-keepassxc)
asking your unlocked KeePassXC. And it adds an `includeIf` rule, so a repo whose
remote is on that host commits with that account's identity. In KeePassXC:

1. Settings → Browser Integration → enable it.
2. With the database unlocked, run once: `git-credential-keepassxc configure`.
3. One entry per host: URL `https://<host>`, the username, and a **personal access
   token** as the password. A login password works for git over HTTPS, but not for
   the APIs `profile repos` uses. Scopes: GitHub `repo`; GitLab `read_api`,
   `read_repository`, `write_repository`; Gitea `read:repository`,
   `write:repository`, `read:user`.

Tokens reach `curl` on stdin, never on a command line.

## One repo, several origins: `tracker sync`

For a repo that lives on several hosts (GitHub, gitlab.com, a company GitLab, a
university GitLab), where some work must stay off some of them.

One **canonical branch** holds everything; you work and commit there. Each origin
receives a **projection** of it: the same history with the paths hidden from that
origin filtered out. Collaborators commit to their origin as usual; their commits are
pulled back into the canonical branch and flow on to every other origin that may see
what they touched.

Visibility is declared in `.sync.toml`, committed on the canonical branch (and never
visible to any origin, since it names what is hidden):

```toml
branch = "main"

[remotes.github]
url  = "git@github.com:you/project.git"
sees = []                          # unlabelled paths only

[remotes.uni]
url  = "git@gitlab.uni-greifswald.de:you/project.git"
sees = ["uni"]

[remotes.bitspark]
url  = "git@gitlab.bitspark.de:you/project.git"
sees = ["uni", "bitspark"]

[hide]                             # glob = label; trailing "/" = whole directory
"bitspark/"      = "bitspark"
"notes/private/" = "private"       # no remote sees "private": it never leaves
```

A path is visible to an origin iff the origin `sees` every label whose glob matches
it. Globs match the full path from the repo root (`*` stays within one directory;
use `**/*.key` for "any depth").

```bash
tracker sync init                  # write a template .sync.toml; edit, then commit it
tracker sync status                # what would happen (fetches, changes nothing)
tracker sync run                   # pull collaborators' commits, push every projection
tracker sync run --remote uni      # just one origin
```

**What happens on `run`:**

- **Pull.** Each new commit on an origin is replayed onto the canonical branch with
  a three-way merge, so hidden files are carried through untouched. It keeps the
  collaborator's authorship and gains a `Sync-Origin: <remote> <sha>` trailer. That
  trailer stays in the canonical repo and is stripped from every projection.
- **Push.** Each origin gets its projection. Projection is deterministic and
  history only ever grows: no push is forced, and a collaborator's commit stays in
  their origin's history. A commit that touches only hidden paths simply doesn't
  appear there.

**Where it stops and asks you instead of guessing:**

| Situation | What to do |
|---|---|
| A collaborator's commit conflicts with your canonical work | `tracker sync resolve <remote>`, fix, `git add`, `tracker sync resolve --continue` (or `--abort`) |
| A collaborator created a file in a path hidden from them | Same `resolve` flow, or change the manifest |
| One of your commits touches both visible and hidden paths, so its message may describe hidden work | `tracker sync message <commit> --remote uni "Extend the paper"` (or `--shared`); or set `mixed_messages = "generic"` / `"keep"` on that remote; or `--trust-messages` |
| An origin already had history this repo never pushed | Not supported yet (see limits below) |

**Guard.** `tracker sync run` installs a `pre-push` hook that refuses a direct
`git push` of the canonical branch to any origin in the manifest. If you already have
a pre-push hook, it is left alone and a warning is printed; add the guard to it by
hand.

**Keep with your private canonical remote.** Message overrides are git notes; push
them to wherever the canonical branch lives, with
`git push <private-remote> refs/notes/sync-messages`. The canonical→origin map in
`.git/tracker-sync/` is only a cache. On a fresh clone it is rebuilt by recognising
the earlier projections.

**Current limits:**

- Hiding is by whole path; parts of a shared file cannot be hidden.
- An origin whose existing history was not produced by `tracker sync` cannot be
  adopted yet. Start it from an empty branch.
- A collaborator's merge is taken in as one commit, credited to whoever made the
  merge.
- A collaborator force-pushing over history already synced is not handled.
- Once pushed, history is permanent. Tightening the manifest hides paths from the
  next commit on; it cannot recall anything already pushed.

## Design invariants it honours

From the split-attention-agents blueprint (each a checkable predicate):

- **I1 — conserved identity:** χ is a weighted-graph invariant, unchanged under
  relabelling. *(tested)*
- **I2 — never-resetting count:** each tracked act (index, χ recompute, run)
  increments a monotone counter `m`; nothing decrements it. *(tested)*
- **I3 — search-not-fetch:** every `sense`/`ask` answer comes from a fresh
  `purpose ask` against the current index, never a stored value.
- **I4 — exclusive phases:** constructing (re-indexing / recomputing χ) and
  committing (answering) never share an instant.

## Status

Working: `init`, `add`, `list`, `sense`, `ask`, `drift`. `run` is stubbed with
graceful degradation pending the network-yield CLI interface. Federation-level χ(Σ)
(the group's own direction) is the next milestone.
