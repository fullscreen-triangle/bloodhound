# Repo-Federation Tracker — Design Document

**Status:** Design (no implementation yet)
**Author target:** Kundai Sachikonye
**Date:** 2026-07-23
**Scope decision:** New Rust CLI, installable per-project, that **shells out** to
`purpose` (search) and the network-yield CLI (execution). It reimplements neither;
it composes them and adds the missing middle layer — **superb tracking of a
federation of repos, each carrying its conserved sense/goal.**

---

## 1. What this tool is (and is not)

A research group has many repos. This tool is a **glorified, sense-aware GitHub
search + federation tracker**: it does not merely track commits, it tracks the
*conserved sense/goal* of each repo, keeps that current as the repos churn, answers
cross-repo questions cheaply, and can hand specific work to persistent agents that
run code in the cloud.

It is **not** a search engine (that is `purpose`) and **not** a scheduler/executor
(that is the network-yield CLI). It is the **society layer** that sits above both.

### The three organs (one paper each, one tool each)

| Organ | Tool | Paper realised | Responsibility |
|---|---|---|---|
| **Search** | `purpose` (installed, v0.1.0) | Split-attention agents §search-not-fetch (Inv. 3) | Index a repo's self-graph; answer "what/where/sense" by *searching, not fetching*. |
| **Execution** | network-yield CLI (in dev) | Network Yield & Computing Allocation | Route compute tasks; each allocated process is a persistent goal-directed agent (occupied-not-waiting, goal succession, summons its own capacity). |
| **Tracking** | **this tool** | Contact-graph foundation (T1/T4) + LBA/split-attention society (T6/T7) | Federate repos; hold each repo's conserved invariant χ; coordinate the persistent agents across the group. |

**Rule the tool honours:** execution and search are *dependencies it calls*, not
features it rebuilds. Its own theorems are T1 (per-repo conserved identity χ),
T4 (never-resetting committed count = tracked history), and T6/T7 (society of repos
+ group-level invariant).

---

## 2. Theory → mechanism map (nothing decorative)

Every construct the tracker maintains is a named result from the papers, so the
implementation stays faithful and checkable.

| Paper construct | Concrete role in the tracker |
|---|---|
| **Self-graph** Γ = (V,E,w) | The repo's structure. **Already built by `purpose index`** — symbols/files/headings as vertices (10,256 in bloodhound). The tracker never builds this itself; it reads `.purpose/index.json`. |
| **Character invariant** χ = min over partitions of the cut-residual (T1) | The repo's conserved **sense/goal**. Stable under relabelling and small commits; it is *what does not change* as the repo churns. This is the headline quantity the tracker reports. |
| **Floor** β > 0 (T0, derived from non-completability) | Guarantees χ > 0: a repo with any structure has a positive, non-collapsible sense. Also the resolution below which the tracker refuses to claim a distinction. |
| **Individuation by negation** (T1) | A repo is characterised against *the rest of the federation* — its sense is partly "what the other repos are not." Powers cross-repo differentiation. |
| **Monotone committed count** m (T4) | The repo's **tracked history** — never resets. Maps onto git history + the tracker's own recorded observations. An authored copy of a repo is a distinct tracked individual. |
| **Search-not-fetch** (T5, Inv. 3 + Realisation note) | The tracker answers every "sense/where" query by invoking `purpose ask` against the *current* index — never from a stored summary. The split-attention paper's Realisation note (lines 1467–1473) describes `purpose` almost verbatim: "a deterministic index … that returns a ranked context slice per query … holds no domain answers, only the means to search for them." |
| **Society graph** Σ (T6) | The **federation** — one vertex per repo, edges = inter-repo relatedness. Σ is itself a contact graph, so T0/T1/T4 apply one level up. |
| **Society invariant** χ(Σ) (T6/T7) | The **group's** research direction — realised by no single repo. "What is this lab actually about" ≠ any one repo's sense. |
| **Water-filling / attention price** p* (T2) | When the tracker must divide a finite budget (re-index this repo? refresh χ? launch a run?) across many repos, it allocates by water-filling with a single price — *conditional on concave returns*, flagged as environmental. |
| **Persistent goal-directed agent** (network-yield §agents) | A tracking agent per repo: occupied (re-indexing/watching), succeeds to a fresh goal on completion, gives a fresh answer each interaction because its state advanced, and **summons compute by raising its price when it stalls** (e.g. "this repo needs a test run"). Execution handled by the network-yield CLI. |

---

## 3. Object model

```
Federation
 ├── registry            (persisted: which repos, where, remotes)
 ├── χ(Σ)                society invariant  (T6/T7)  — derived, cached, regenerated on member change
 └── Repo[]              one tracked repo each
       ├── path / remote
       ├── self-graph    = .purpose/index.json  (owned by `purpose`, not us)
       ├── χ             character invariant (T1) — derived from the index, cached
       ├── m             committed count (T4) — monotone, persisted, never decremented
       ├── salient       high-centrality symbols/headings (the "sense" surface)
       └── agent-handle  optional: a persistent goal-directed agent (network-yield)
```

**Persisted tracker state** lives in a federation-root file (proposal:
`.tracker/federation.json` + per-repo `.tracker/<repo>.json`), analogous to how
`purpose` owns `.purpose/index.json`. The tracker **owns χ, m, salient, and the
registry**; it **borrows** the self-graph from `purpose` and **borrows** execution
from network-yield.

### χ — the character invariant, concretely (decision: structural, no LLM)

Computed deterministically from the `purpose` self-graph:

1. Load `.purpose/index.json` (run `purpose index` first if missing/stale).
2. Build the weighted graph: vertices = indexed symbols/headings; edges = co-occurrence / containment / reference proximity; weights ≥ β.
3. **χ = minimum cut-residual over nontrivial partitions** (T1, Def. character invariant) — the least total cost of splitting the repo into pieces, in the currency of its own separations. Positive by T0, non-local by T1 (a region, never one symbol).
4. **Salient surface** = the vertices/blocks realising or bordering that minimum cut + highest-centrality headings — this is the human-readable "sense/goal" the query layer narrates *by search*, never by storing prose.

χ is **cached** but treated as *fetch-forbidden for answers*: a `sense` query
re-derives the salient surface via `purpose ask` against the live index (T5), so a
report always reflects the repo as it is now. The cached χ scalar is only a
change-detector (did the sense move?), not the answer itself.

> Open modelling choice deferred: exact edge construction from the index
> (co-occurrence window vs. import graph vs. heading tree). The paper fixes χ's
> *form* (min cut-residual); the edge model is an implementation degree of freedom
> to pin during build, with the numerical-witness harness (below) validating
> invariance under relabelling.

---

## 4. The four implementation invariants (from split-attention §blueprint)

The tracker is a faithful runtime iff it preserves these. Each is a checkable
predicate with the theorem that certifies it. Three are **unconditional**
(agent-structural); the attention scheduler is **environment-conditional**.

| # | Invariant | Predicate the tracker must satisfy | Certified by |
|---|---|---|---|
| **I1** | **Conserved identity** | Under any internal update that preserves separations+costs (re-index that only relabels, representation change), χ(repo) is unchanged; χ is scene-independent (same across who queries it). | T1 (Identity) |
| **I2** | **Never-resetting count** | m increments on every committed tracked act (observation, re-index commit, run); never decremented; persisted across sessions. No `undo`/restart lowers m. A cloned repo starts at m=0 → distinct tracked individual. | T4 (History) |
| **I3** | **Search-not-fetch** | No `sense`/`where` answer is emitted without invoking a fresh `purpose ask` (≥1 search act) against the *present* index. There is no stored-answer read-out path. | T5 (Recognition=Search) + Realisation note |
| **I4** | **Exclusive phases** | At any instant a repo's agent is either *constructing* (re-indexing / recomputing χ; emits no answer) or *committing* (answering / launching a run; no graph update). The two instant-sets are disjoint. | T3 (Alternation) |
| **(cond.)** | **Attention scheduler** | Dividing a finite budget across repos/tasks is water-filling at a single price p*. **Conditional on concave returns** — a fact about the workload, not the tool. Off-concavity the scheduler is a heuristic, not optimal; the tool must not dress the conditional guarantee as unconditional. | T2 (Water-filling) + Ax. concave |

**Negative controls to ship** (mirroring the paper's 3/3 detected violations): a
fetch-cache path (must be rejected by I3), a count-rollback (rejected by I2), a
construct-and-answer-in-one-instant (rejected by I4). A green suite is only
meaningful if these red cases are caught.

---

## 5. Composition contract — exactly which calls go where

### 5.1 To `purpose` (search organ) — **confirmed interface**

Installed at `~/.cargo/bin/purpose`, v0.1.0. Single-repo today (`--root` targets
others); cross-repo is the tool's own job (§5.3).

| Tracker need | `purpose` call | Notes |
|---|---|---|
| Build/refresh a repo's self-graph | `purpose index [--root <repo>]` | Produces `.purpose/index.json`. Run when missing/stale (I3 freshness). |
| Answer a sense/where query | `purpose ask "<q>" [--root <repo>]` | Returns ranked `file:line [kind] name` + snippet slice (~200 tok). **This is the I3 organ.** |
| Inspect compiled query (debug) | `purpose ask "<q>" --dry-run` | Emits the vaHera compose fragment. |

**Known constraints (verified from source `purpose-domains-codebase/src/lib.rs`):**
- Ranking is raw substring: +3 if term in symbol name, +1 in snippet, top-20. No
  idf, no config, no synonym map. → the tracker must issue **distinctive
  multi-term** queries and may need to fan out several `ask` calls and merge.
- No config surface at all (no `.purpose/config.toml`, no env). Extensions, ignore
  list, ranking weights are hardcoded. → tracker cannot tune `purpose`; it adapts
  around it or (future) upstreams a change.
- Index is a snapshot → tracker owns staleness detection (git HEAD moved →
  re-index before answering, to honour I3).

### 5.2 To the network-yield CLI (execution organ) — **interface, binary in dev**

Not yet on PATH. The tracker targets the paper's public surface, not a pinned
binary name. Contract (to finalise when the binary stabilises — **TODO: confirm
actual argv**):

| Tracker need | network-yield concept | Paper anchor |
|---|---|---|
| Launch specific code for a repo (Codespaces/cloud) | Submit a **task** with a completion target; the allocated process is a persistent goal-directed agent | §agents, Def. process agent |
| A stalled tracking need summons compute | Agent **raises its own price** (rising separation cost → clearing price → capacity routed) | Prop. agent-initiated contact; Thm. liveness Step 3 |
| Divide compute across many repo-tasks | **Yield market / backpressure routing** clears at compute-tick granularity | Thm. Three-way Equivalence |
| Persist a run's history | **Monotone committed-step counter** M (non-forgeable) — reconcile with tracker's own m (I2) | Thm. incorruptibility (iv), Cor. life-history |

**Graceful degradation (mirror `purpose`'s missing-binary handling):** if the
network-yield binary is absent, the tracker runs fully as a read-only
tracker/search federation; `run`/execution commands return a clear "execution
organ not installed" status rather than failing the whole tool. Tracking (χ, m,
sense, federation) never depends on execution being present.

### 5.3 The tracker's own layer (neither borrowed)

- **Registry**: named repos, remotes, local paths. This *is* `purpose`'s
  "forthcoming multi-repo Tool B Layer 2" — the tracker fills it.
- **χ computation** (§3) from the borrowed index.
- **Federation graph Σ** and **χ(Σ)** (T6/T7): edges between repos from shared
  salient vocabulary / cross-repo `purpose ask` hits; χ(Σ) = min cut-residual over
  the repo-graph. "Which repos form a coherent subproject" = a partition of Σ.
- **Committed count m** and history (I2).
- **Water-filling scheduler** (T2, conditional) deciding what to do next across
  repos under a budget.

---

## 6. CLI surface (proposed — for the doc, not final)

```
tracker init                         # create federation root (.tracker/)
tracker add <path|remote> [--name]   # register a repo; runs `purpose index`
tracker list                         # repos + current χ + last-tracked m
tracker sense <repo> ["<question>"]  # I3: fresh `purpose ask`; reports salient sense.
                                     #   no arg → the repo's standing sense/goal
tracker ask "<question>"             # cross-repo: fan `purpose ask` over the
                                     #   federation, merge+rank, cite repo:file:line
tracker drift <repo>                 # did χ move since last observation? (change-detect)
tracker federation                   # χ(Σ): the group's direction; sub-project clusters
tracker run <repo> "<goal>"          # hand a goal to network-yield (execution organ)
                                     #   degrades to "not installed" if binary absent
tracker status                       # per-repo m, phase (I4), agent handles, scheduler price
```

Installable per-project like `purpose`: `cargo install --path <crate>`.

---

## 7. Numerical witness (ship with the tool, like both source papers)

Every source paper pairs claims with an exhaustive check; the tracker should too:
- **I1**: χ unchanged to machine precision under random relabelling of a repo's
  index; non-local (min partition is multi-block, never a singleton).
- **I2**: m strictly monotone across simulated sessions; clone starts at 0.
- **I3**: every `sense`/`ask` answer traces to a `purpose ask` invocation
  (log the subprocess call); fetch-cache negative control is detected.
- **I4**: construct/commit instant-sets disjoint; the multitask negative control
  is detected.
- **T6/T7**: χ(Σ) ≥ β, non-local across repos; a known 2-cluster federation
  fixture resolves to the 2-block partition.

---

## 8. Build sequence (after this doc is approved)

1. **Scaffold** the Rust CLI crate; `init`/`add`/`list` + registry persistence.
2. **χ from index**: load `.purpose/index.json`, build weighted graph, compute
   min cut-residual + salient surface. Ship the I1 relabelling witness.
3. **`sense` / `ask`**: shell to `purpose ask`, enforce I3 (freshness + no fetch),
   cross-repo fan-out + merge for `tracker ask`.
4. **m + history** (I2) with rollback negative control; **phases** (I4).
5. **Federation Σ / χ(Σ)** (T6/T7) + `federation`, `drift`.
6. **Execution bridge**: once network-yield argv is pinned, `run` + price-summon;
   until then, the graceful-degradation stub.
7. **Water-filling scheduler** (T2) across repos, clearly labelled conditional.

---

## 9. Open questions to resolve before / during build

1. **Edge model for χ** (§3): which relation defines index-graph edges? (Pin during
   step 2; validate via the I1 witness.)
2. **network-yield argv** (§5.2): exact CLI to invoke — confirm when the binary
   stabilises. Where does its crate live? (paper: `distributed/pylon/panthera/…`).
3. **m vs network-yield's M**: two monotone counters (tracker I2, execution
   incorruptibility). Reconcile: is the tracker's m a superset that absorbs run
   events, or are they separate ledgers linked by reference?
4. **Where `.tracker/` lives** for a *federation* spanning many repo roots — a
   central home dir vs. a designated "hub" repo.
5. **`purpose` ranking weakness** (§5.1): accept multi-query fan-out as the
   workaround, or (later) upstream idf/synonym support to `purpose` so the tracker
   gets sharper slices for free?
```
