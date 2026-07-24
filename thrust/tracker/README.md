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
