# Cross-Repo Confluence — Design Document

**Status:** Design (no implementation yet)
**Author target:** Kundai Sachikonye
**Date:** 2026-07-26
**Scope decision:** A new capability layer over the existing tools that treats a
federation of *N* repositories as **one queryable, composable, runnable library**.
It adds three things on top of what already exists: (1) **cross-repo
correspondence** — finding where the same idea lives in more than one repo, and
where an idea in one repo has no analog in another; (2) **pipeline composition** —
plucking specific methods from wherever they live and wiring them into a runnable
program; (3) **execution + coherence** — running each step of that program **sealed
in its own container** (the repo's own devcontainer) in GitHub Codespaces, passing
data across container boundaries, and checking, with **wind-tunnel**, whether the
assembled units actually cohere toward the asked-for goal.

It reimplements none of the underlying organs. It composes `purpose` (search),
the repos' Codespaces (execution), and `wind-tunnel` (coherence analysis).

---

## 1. The problem, stated exactly

A research group has 67+ repos, mostly science. Three exemplars:

| Repo | Discipline |
|---|---|
| **lavoisier** | mass spectrometry |
| **gospel** | genomics |
| **borgia** | cheminformatics |

These disciplines **overlap conceptually but the code does not overlap**. There is
no shared import, no common package, often not even a common language. As a direct
consequence there are two invisible pathologies across the corpus:

- **Duplicated ideas** — the same concept independently reimplemented in, say,
  borgia and lavoisier, neither aware of the other.
- **Orphan ideas** — a method that exists only in gospel that *would be useful* in
  lavoisier, but was never carried over.

Both are invisible at the file level (different names, different languages, no
citations between them) and visible only at the level of **what the code is
*about*** — which is precisely what the `purpose` graph and the character invariant
χ already capture. This is the entire justification for tracking *sense* rather than
commits: commit history can never surface a duplicated or orphan idea across two
unrelated repos; a sense-graph can.

### The cost being removed

Today, to use these tools effectively a person must, **per repo**: clone → install
→ understand → use. For 67 repos this tax is prohibitive; the corpus is effectively
unusable as a whole. Confluence removes the per-repo tax and makes the federation
behave as a single library with a single query surface.

---

## 2. One interface, two priors (not two modes)

An earlier framing split "uninformed user" and "developer" into two tools. **That
split is rejected.** The developer is *the same user with more prior knowledge*, not
a different tool:

> "The developer is just like an uninformed user, except that they have an idea of
> what the code in each repo does."

Therefore there is **one interface** and **one flow**. The only difference is what
the human already knows, which changes *how they phrase the question*, not *what the
system does*:

```
abstract question
      │
      ▼
 correspondence      ← find the relevant ideas, wherever they live across N repos
      │                (and surface duplicates / orphans encountered on the way)
      ▼
 composition         ← assemble the relevant methods into a runnable pipeline
      │
      ▼
 execution           ← run just those methods, in the repos' own Codespaces
      │
      ▼
 coherence           ← wind-tunnel: do the assembled units cohere toward the goal?
      │
      ▼
 result + regime map
```

An uninformed user asks *"which of my tools can align a spectrum to a candidate
structure?"* and never learns which repo answered. The developer asks *"is
borgia's fragment scorer really the same idea as lavoisier's peak matcher?"* — a
sharper question into the same machinery. Same pipes.

---

## 3. Where this sits relative to the existing organs

| Organ | Tool | Responsibility | Confluence's use of it |
|---|---|---|---|
| **Search** | `purpose` (installed) | Index each repo's self-graph; answer what/where/sense by searching, not fetching. | The **substrate.** Every repo's ideas come from its `.purpose/index.json`. Confluence never re-indexes. |
| **Tracking** | repo-federation tracker (built) | Hold each repo's χ and monotone history; federate repos. | Confluence is the **query + composition layer above the tracker's federation.** The tracker says *what each repo is*; confluence says *how they correspond and compose*. |
| **Execution** | GitHub Codespaces (per repo, exists) | Actually run code. | Confluence **directs** a repo's Codespace to run specific methods and emit traces. It does not run code itself. |
| **Coherence** | `wind-tunnel` (external, exists) | Consume the `purpose` graph + `{t,state[]}` traces → a regime map (R_dyn, semantic floor, holonomy, decoherence, contribution). | Confluence feeds wind-tunnel the **composed pipeline's** graph + the traces the Codespace produced, to test the assembled system that exists in no single repo. |

**Rule honoured (same as the tracker):** search, execution, and coherence are
*dependencies called*, not features rebuilt. Confluence adds only the missing
middle: correspondence and composition.

### Two facts that anchor the whole design

1. **wind-tunnel does not run code.** Confirmed from its README: its dynamic phase
   consumes *pre-recorded* JSONL traces (`{"t": <float>, "state": [<float>...]}`,
   one file per unit, filename stem = unit id). "Run your system under load and
   collect traces." So *execution* and *coherence* are strictly separate problems.
2. **Codespaces is the execution organ.** Every repo already has one. The Codespace
   is what runs specific code and — when instrumented — is what *produces the traces*
   wind-tunnel then reads. network-yield routing is explicitly **out of scope** for
   this design; the only execution target here is the repo's own Codespace.

---

## 4. The unit of correspondence is the *idea*, not the file

Confluence's index is **not** file-centric. The vertex is a **method/idea**: a
`purpose` symbol (its name, kind, signature, snippet, containing file, and its
χ-region — the part of the repo's sense-graph it lives in). Correspondence is a
relation *between* such vertices *across* repos:

- **`same_as`** — two ideas in different repos are the same concept (duplication).
- **`fits`** — an idea in repo A has no analog in repo B but *would be useful* there
  (orphan / opportunity).
- **`composes_with`** — output of one idea can feed the input of another (the edge
  the pipeline builder walks).

This is a society graph (T6) one level below the tracker's: the tracker's Σ has one
vertex per repo; confluence's correspondence graph has one vertex per *idea* and
edges that cross repo boundaries.

---

## 5. Correspondence engine — LLM adjudicates, structure retrieves

**Decision: LLM adjudicates candidate pairs.** The pathology is a *cross-vocabulary*
problem — borgia's Rust term and gospel's Python term for the same concept share no
tokens — so pure structural matching under-recalls, and pure embedding matching
gives no defensible reason for a match. The chosen pipeline separates **retrieval**
(cheap, deterministic, LLM-free) from **adjudication** (expensive, LLM, on the small
candidate set only):

```
 stage 1  RETRIEVE candidates   (no LLM, over all N repos)
          structural signals from purpose: symbol kind, signature arity/shape,
          snippet term overlap, χ-region role (salient? cut-side?).
          embedding similarity of (name + signature + snippet) via the existing
          HF/Ollama layer, cross-repo nearest neighbours.
          → a bounded set of candidate pairs (A.idea, B.idea), ranked.
                              │
 stage 2  ADJUDICATE pairs     (LLM, only on the candidate set)
          for each candidate the LLM answers, with justification:
            · same_as?  (is this the same idea, different vocabulary?)
            · fits?     (useful in the other repo but absent there?)
          → a verdict + a one-paragraph reason per pair.
                              │
 stage 3  RECORD               verdicts stored as correspondence edges,
          each stamped with the model, prompt version, and the retrieval
          signals that proposed it — so a verdict is reproducible-as-audited
          even though the LLM step is not bit-reproducible.
```

**Why not LLM-only:** at 67-repo scale, adjudicating all pairs is quadratic and
ruinous. Retrieval bounds the LLM to the plausibly-related few. **Why not
structure-only:** it cannot see "same idea, different words," which is the whole
value. The LLM is the *judge*, never the *retriever* — it never sees the 67 repos,
only the pairs structure already flagged. This mirrors the AI path already in
repo-lens (generate → validate against a ground-truth compiler → repair): here the
LLM proposes a *semantic* verdict and the *structural retrieval* is the ground that
constrains it.

### Honesty flags

- LLM verdicts are **non-reproducible bit-for-bit**; we record inputs + model +
  prompt version so a verdict is *auditable and re-runnable*, not deterministic.
- A `same_as` verdict is a **claim, not a proof.** The UI must present it as an
  LLM judgement with its justification, never as ground truth. Duplicated-idea
  reconciliation is a human decision the tool *informs*.
- Embedding + LLM steps require the model layer to be reachable (HF key server-side,
  or local Ollama). With neither, confluence **degrades to structural-only
  correspondence** and says so — the same graceful-degradation discipline the
  tracker uses when `purpose`/execution are absent.

---

## 6. Composition — from ideas to a runnable pipeline

Once correspondence has found the relevant ideas for a question, they must become a
program. The composition layer is a small, explicit **pipeline IR** — not free-form
code generation:

```
Pipeline
  steps: [ Step ]
  edges: [ Edge ]         composes_with data hand-offs between steps
Step
  repo         which repo the method lives in
  symbol       the purpose symbol (method) to invoke
  image        the container this step runs in — the repo's own devcontainer by
               default (see §7); each step is sealed in its repo's environment
  entry        how the container calls it (module path + callable + arg binding)
  inputs       bound from prior steps' outputs or the user's arguments
  emits_trace  whether this step is instrumented to produce {t,state[]}
Edge
  from → to    producer step → consumer step
  contract     the serialization format of the data crossing the boundary
               (schema + encoding); this is where cross-language hand-off is
               made explicit and where composition can legitimately fail
```

- For the **uninformed user**, the pipeline IR is generated from the abstract
  question (LLM drafts it) and **validated** before it can run — the same
  empty-dictionary principle: a real validator, not the model, is ground truth. A
  step naming a symbol that does not exist in that repo's `purpose` index is
  rejected and repaired.
- For the **developer**, the same IR is authored/edited directly (they know the
  methods). Same object, different author.

The IR is deliberately *runnable-per-step*: each step maps to one method invocation
in one repo's own container (§7). This is what makes "extract just the right methods
and run them" concrete — the pipeline is a list of `(repo, method, wiring)` steps
plus the `contract` on each edge that says how one step's output becomes the next
step's input across the container boundary.

---

## 7. Execution — per-container steps in Codespaces

**Decision: one container per repo/step.** A cross-repo pipeline mixes environments
that cannot coexist in one process — borgia is Rust, gospel and lavoisier are Python
with conflicting dependency trees; there is *no single environment that satisfies all
of them*. So each step runs **sealed in its own container, built from that repo's own
devcontainer image**. A Codespace is already a container (built from a devcontainer),
so this is not a new substrate — it is using Codespaces' native isolation as the
step boundary: borgia's step runs in borgia's container, gospel's in gospel's, each
with exactly the toolchain that repo needs and nothing else.

This turns the pipeline from "in-process calls inside one environment" into **a
dataflow of isolated containers that pass data across their boundaries.** Each
`composes_with` edge is a real serialization hand-off (§6's `Edge.contract`), not a
function call. That is a stronger isolation guarantee and it matches how a human
would actually chain two of these tools: run tool A, take its output file, feed it to
tool B.

```
 for each Step in Pipeline (topological order):
   1. bring up the step's container   from the repo's devcontainer image
                                       (gh codespace / GitHub API, user token)
   2. materialise inputs into it       bound args + upstream step outputs,
                                       decoded per the incoming edge's contract
   3. run the method, wrapped by the   tracing adapter (below)
   4. capture:  result  → encoded per the outgoing edge's contract
                trace   → traces/<step>.jsonl   ({t, state[]} lines)
   5. tear the container down          (isolation ends at the step boundary)
 collect all traces/*.jsonl into one directory for wind-tunnel (§8)
```

- **Container = repo devcontainer by default; explicit `image` override allowed.**
  The default is the repo's own `.devcontainer` image, so a step inherits exactly the
  environment the repo already defines. A step may name an explicit `image` when the
  method needs something the devcontainer does not provide. (This is the "repo
  default, image override" model.)
- **Trigger path:** the GitHub Codespaces API / `gh codespace`, using the same
  personal-access token repo-lens already takes. (network-yield routing is out of
  scope by decision.)
- **The cross-container data contract** (`Edge.contract`) is where composition can
  legitimately fail: borgia's output must serialize to something gospel can read. The
  contract makes that boundary explicit and checkable *before* a run — a mismatched
  contract is a composition error surfaced at validation, not a mysterious runtime
  crash. This is the same boundary a person hits piping one tool into another; the IR
  names it instead of hiding it.
- **The tracing adapter** produces wind-tunnel input: a small, language-appropriate
  wrapper the runner injects around the invoked callable that records
  `{"t": <float>, "state": [<float>...]}` per observed step into
  `traces/<step>.jsonl`. Because each step is its own container, its trace is emitted
  at a clean boundary with no cross-step contamination — exactly wind-tunnel's
  one-file-per-unit shape. A method that cannot be reduced to a state vector runs but
  emits no trace (dynamic analysis simply skips it).

### Honesty flags

- Codespaces/containers have **cost, cold-start latency, and quotas.** One container
  per step means more boots than one-Codespace-per-run; the runner should reuse a
  warm container across *consecutive steps of the same repo* rather than one per step,
  and must surface cost/time.
- **Cross-container hand-off is a real serialization boundary.** If borgia's output
  has no faithful encoding gospel can consume, the pipeline *cannot* compose there —
  and that is correct information, not a bug. Validated at the `contract`, up front.
- Arbitrary methods are **not always callable in isolation** (hidden setup, fixtures,
  data). The IR's `entry` binding is where this is made explicit; steps that cannot
  be isolated fail loudly with the reason, not silently.
- **Instrumentation is opt-in per language/repo.** Without an adapter for a repo's
  language, a step runs but is trace-less → that step contributes to the *result*
  but not to the *coherence* analysis. This is the honest boundary and must be shown.

---

## 7.5 The orchestrator is the seam, not a separate unit

There is a temptation to draw an "orchestrator/AI" as a third autonomous component
that decides what to run and fires it off. **That framing is rejected**, for the same
reason the two-view split was (§2): it would take the human *out* of the loop that
already exists.

The loop already has both halves, and they are two sides of one coin:

- **After a repo is analysed → there are results** — χ, the salient surface, the
  correspondence verdicts, the drafted pipeline IR. This is the machine-produced side.
- **Then there is human direction** — the person looks at those results and points:
  *"yes, run that," "wire it differently," "extract this method."*

The "orchestrator" **is those two facing each other.** It is not a daemon that
supplants the human; it is the analysis-produces-a-runnable-thing side meeting the
human-points-at-it side. Concretely, **the orchestrator is the repo-lens notebook loop
itself** (§9): a results cell is produced → the human acts on it → a dispatch cell
fires. Nothing new to host; the seam is already the cockpit.

### GitHub-native dispatch (decision)

When the human acts, execution is dispatched **GitHub-natively** — the run driver is a
**GitHub Action** (`workflow_dispatch`) living in the target repo, not a backend the
user hosts. This leans fully into "use the GitHub setup": the Action already has native
access to the repo, its Codespaces/devcontainer, and the token; it runs *where the code
and the compute already are*. The notebook emits the approved pipeline IR + the human's
go-ahead into a `workflow_dispatch` event; the Action brings up the per-step containers
(§7), runs the methods, and reports results + collected traces back. This completes the
framework's own metaphor: **Codespaces is the VM, the devcontainer is the OS image, the
Action is the dispatch driver, and the orchestrator is the analysis↔direction seam** —
no separate agent process is introduced.

| Framework concept | Provider |
|---|---|
| Compute substrate (the VM) | GitHub Codespaces |
| OS image per step | the repo's devcontainer (§7) |
| Dispatch / run driver | **GitHub Action** (`workflow_dispatch`), native to each repo |
| Orchestrator | **the analysis↔human-direction seam** — the notebook loop, not a daemon |

**Honesty flag.** "GitHub-native, no backend to host" is true of *dispatch and
execution*; the *adjudication* step (§5) still needs a model endpoint (the existing
`/api/repo-lens/ai` proxy or local Ollama), because provider keys must not sit in a
public Action. So the split is: correspondence/composition run where the notebook + AI
proxy run; execution runs in GitHub. The seam spans both, which is exactly why it is a
seam and not a single service.

---

## 7.6 Deployment shape — a downloaded local engine, not a hosted backend

The web tool is deliberately **static** (the existing thrust Next.js site). Rather than
stand up a hosted Rust backend, the parts that a browser tab *cannot* do are shipped as
a **native Rust engine the user downloads and runs locally**, exposing a localhost API
the static tool talks to (the Jupyter / Ollama pattern):

```
  Browser tab (static thrust site)
        │  fetch / WebSocket → localhost:<port>
        ▼
  bloodhound engine (downloaded native Rust binary)
        ├── reads local repo clones            (filesystem — a tab cannot)
        ├── shells to purpose / tracker / wt    (subprocess — a tab cannot)
        ├── holds the GitHub token locally      (never touches our infra)
        └── dispatches workflow_dispatch via gh (§7.5)
```

**Why not WASM-in-tab for the engine:** a browser WASM sandbox has no filesystem, no
subprocess, no arbitrary sockets — and confluence's core is exactly filesystem +
subprocess (`purpose`/`tracker`/`wt`) + `gh`. So a pure-WASM tab can be a *calculator*
but not the engine. **Why not a hosted backend:** it would put us in the business of
holding user tokens and running user code on our infra; the local engine keeps both on
the user's machine, strengthening the privacy story repo-lens already promises.

**WASM still earns a place — for compute, not access.** The pure-math kernels (χ
min-cut-residual, the wind-tunnel *static* regime computation, embedding similarity)
compile to WASM and run **in the tab** on data the engine already delivered, for
instant interactivity with no round-trip. Division of labour: **WASM for computation
on data-in-hand; the native engine for anything touching the machine, the tools, or the
network.**

**Honesty flags.**
- The download is a real adoption tax vs. "just visit a URL." Accepted because the
  audience already clones repos; one binary is far cheaper than confluence's per-repo
  tax (§1). An uninformed drive-by user is *not* the near-term target of this shape.
- HTTPS-page → HTTP-localhost mixed-content, CORS, and the port handshake are real
  plumbing (solved by Ollama/Jupyter/wrangler, but not free).

**Open (engine-spec, not architecture) — deferred by decision:**
- **Full engine vs. thin bridge** — whether the binary *is* confluence (registry +
  correspondence + orchestration inside it) or only exposes FS/subprocess/`gh` while
  correspondence/IR logic stays in JS/WASM in the tab.
- **Local clones vs. clone-on-demand** — whether the engine reads repos already on disk
  or fetches them itself via `gh` when a query touches an absent repo.

Both are settled when the engine itself is specced; the architecture above (local native
engine + static tab + WASM compute kernels) holds either way.

---

## 8. Coherence — wind-tunnel on the assembled pipeline

This is *why wind-tunnel belongs at the federation level and nowhere else*: the
composed pipeline is a system that **exists in no single repo**, so no single repo's
tests can vouch for it. Confluence assembles methods from 3 different repos toward
one goal; the live question is whether those units *cohere*.

Two phases, matching wind-tunnel's own two:

- **Static (always available).** Needs only the `purpose` graph of the composed
  pipeline — the union of the involved χ-regions plus the `composes_with` edges.
  Yields tension, R_est, decoherence zones, and a **regime classification** with no
  execution at all. Every composed pipeline gets this for free.
- **Dynamic (when traces exist).** Feeds the `traces/*.jsonl` the runner collected
  to `wt run`, yielding R_dyn(t), holonomy (does the pipeline deviate from the goal
  spec?), and per-step contribution scores (which method actually carried the
  result). Available only for the instrumented steps. Per-container execution (§7)
  pays off here: each step's trace comes from its own sealed container, so the
  `{t,state[]}` files are cleanly per-unit with no cross-step bleed — precisely the
  one-file-per-unit input wind-tunnel expects, and it makes the contribution score
  (ablation of one unit) meaningful because units are genuinely isolated.

The output is a **regime map, not a pass/fail** — consistent with wind-tunnel's
thesis. For a cross-repo pipeline this reads as: *"these methods, assembled toward
your goal, coordinate coherently / show a decoherence zone at step k / step j
contributed nothing."*

---

## 9. What the cockpit shows (repo-lens becomes the confluence notebook)

The existing repo-lens notebook is the natural cockpit; confluence extends it rather
than replacing it. New cell types, all D3-interactive, all appended to the same
growing transcript (the T4/I2 monotone-committed feel already there):

| Cell | Renders | Interaction |
|---|---|---|
| **Correspondence map** | ideas as nodes, `same_as`/`fits`/`composes_with` as edges, colored by repo; duplication clusters and orphan ideas highlighted. | hover → the LLM's verdict + justification; click → the two symbols on GitHub, side by side. |
| **Pipeline** | the composed IR as a left-to-right dataflow of `(repo, method)` steps. | hover a step → its symbol/signature + GitHub link; edit inputs; run. |
| **Regime map** | wind-tunnel output: R_dyn / holonomy / decoherence zones / per-step contribution. | hover a decoherence zone → which steps + why; click a step → the method that owns it. |

The existing χ-bars / salient-treemap / fragment-graph dashboard stays as the
per-repo view; confluence adds the *cross-repo* views above it.

---

## 10. Theory → mechanism map (nothing decorative)

| Construct | Role in Confluence |
|---|---|
| **Self-graph** Γ (purpose) | Source of every idea-vertex. Never rebuilt. |
| **Character** χ / χ-region (T1) | Locates an idea within its repo's sense; the retrieval signal "same structural role." |
| **Individuation by negation** (T1) | An orphan idea is one whose negation-set (what the *other* repos are not) has a gap — the formal shape of "useful here but missing." |
| **Society graph** Σ (T6) | Two levels: tracker's Σ (repo vertices) and confluence's correspondence graph (idea vertices, cross-repo edges). |
| **Society invariant** χ(Σ) (T6/T7) | The lab's actual research direction, realised by no single repo — what "N repos as one" *is*, made computable. |
| **Search-not-fetch** (Inv. 3) | Correspondence retrieval always runs against the *current* purpose index, never a stored summary. |
| **Empty-dictionary / validate-not-memorise** | Pipeline IR and AI-drafted correspondence queries are validated by a real checker (index + IR validator), not trusted from the model. |
| **wind-tunnel regime map** WT(E,Λ)=(R_dyn,S_flat,H,D,δS) | Coherence of the assembled cross-repo pipeline — the test of a system no repo contains. |

---

## 11. Sequencing

The semantic layer is built and proven useful **before** any Codespaces execution,
because each stage is independently valuable and execution is the riskiest, costliest
piece. (Decision: full design doc first — this document — then implement in this
order.)

1. **Correspondence, structural-only.** Cross-repo candidate retrieval over the
   tracker's federation; the correspondence-map cell. Immediately answers "where do
   my ideas overlap / where are the orphans" with no model and no execution.
2. **Correspondence, adjudicated.** Add the embedding retrieval + LLM adjudication
   with recorded, audited verdicts and justifications. This is the killer capability.
3. **Composition.** The pipeline IR + validator + the pipeline cell. "N repos as one
   library" becomes assemblable, still without running anything.
4. **Static coherence.** wind-tunnel static phase over a composed pipeline's graph —
   a regime classification for every assembled pipeline, no Codespace needed.
5. **Execution + dynamic coherence.** The Codespaces runner + tracing adapter +
   `wt run` on collected traces + the regime-map cell. The full "run it and test it."

Each numbered stage ships and is usable on its own. Nothing after stage 3 is needed
for the corpus to already feel like one library.

---

## 12. What this is, and is not

**Is:** a correspondence + composition + coherence layer that makes 67 repos behave
as one queryable, composable, testable library — surfacing duplicated and orphan
ideas, assembling cross-repo pipelines, and testing whether the assembly coheres.

**Is not:** a re-indexer (that is `purpose`), a code runner (that is Codespaces), or
a test executor that runs your code (wind-tunnel runs on *traces*, never on code). It
is also **not a separate orchestrator agent**: the "orchestrator" is the seam where
analysis results meet human direction (§7.5), dispatched GitHub-natively — not a daemon
that runs on its own. It is the confluence — the place the separate rivers of the
corpus are made to meet.
