import Head from "next/head";
import Link from "next/link";
import { useState, useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { probeIterate } from "@/lib/pdsvm";
import { mapQueryToPurpose, suggestRefinements } from "@/lib/intentMapper";
import cytochromeData from "../../public/data/cytochrome.json";

const SEntropyScatter = dynamic(() => import("@/components/charts/SEntropyScatter"), { ssr: false });

// ── State machine states (Blank Screen Paradigm) ─────────────────────────────
// B  → blank  (initial, cursor visible)
// P  → prompt (user typing)
// A  → artifact (results rendered)
// AP → artifact + prompt (refinement)
const STATES = { B: "B", P: "P", A: "A", AP: "AP" };

const EXAMPLE_QUERIES = [
  "Which enzymes metabolize warfarin?",
  "CYP3A4 substrates and drug interactions",
  "Steroidogenesis pathway enzymes",
  "Vitamin D activation and catabolism",
  "Carcinogen bioactivation in smokers",
  "Polymorphic CYPs in pharmacogenomics",
  "Retinoic acid metabolism during embryogenesis",
];

function RelevanceBadge({ level }) {
  const cfg = {
    high:     { label: "High Clinical Relevance",  cls: "bg-red-900/30 text-red-300 border-red-800/40" },
    moderate: { label: "Moderate Relevance",        cls: "bg-amber-900/30 text-amber-300 border-amber-800/40" },
    low:      { label: "Low Relevance",             cls: "bg-surface text-muted border-primary/10" },
    minimal:  { label: "Minimal",                   cls: "bg-surface text-muted/60 border-primary/5" },
  };
  const c = cfg[level] || cfg.low;
  return (
    <span className={`text-xs px-2 py-0.5 rounded border font-mono ${c.cls}`}>{c.label}</span>
  );
}

function SEntropyCoord({ sk, st, se }) {
  return (
    <span className="font-mono text-xs text-primary/80">
      ({sk.toFixed(2)}, {st.toFixed(2)}, {se.toFixed(2)})
    </span>
  );
}

function EnzymeCard({ enzyme, rank, isTop }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: rank * 0.04 }}
      className={`border rounded-xl transition-all duration-200 cursor-pointer
        ${isTop
          ? "border-primary/40 bg-surface hover:border-primary/60 hover:shadow-glow"
          : "border-primary/10 bg-dark/60 hover:border-primary/25"
        }`}
      onClick={() => setExpanded((e) => !e)}
    >
      <div className="flex items-start gap-4 p-4">
        {/* Rank */}
        <div className="w-7 text-center shrink-0">
          <span className={`text-sm font-bold font-mono ${isTop ? "text-primary" : "text-muted/50"}`}>
            {rank + 1}
          </span>
        </div>

        {/* Gene ID */}
        <div className="shrink-0 w-20">
          <div className={`font-mono font-bold text-sm ${isTop ? "text-primary" : "text-muted"}`}>
            {enzyme.id}
          </div>
          <div className="text-xs text-muted/60 mt-0.5">Family {enzyme.family}</div>
        </div>

        {/* Main info */}
        <div className="flex-1 min-w-0">
          <div className="text-sm text-light/90 leading-snug mb-1.5">{enzyme.function}</div>
          <div className="flex flex-wrap gap-2 items-center">
            <SEntropyCoord sk={enzyme.sk} st={enzyme.st} se={enzyme.se} />
            <RelevanceBadge level={enzyme.clinical_relevance} />
            {enzyme.polymorphic && (
              <span className="text-xs px-2 py-0.5 rounded border border-violet-800/40 bg-violet-900/20 text-violet-300 font-mono">
                polymorphic
              </span>
            )}
            {enzyme.inducible && (
              <span className="text-xs px-2 py-0.5 rounded border border-amber-800/40 bg-amber-900/20 text-amber-300 font-mono">
                inducible
              </span>
            )}
          </div>
        </div>

        {/* Expand toggle */}
        <div className="shrink-0 text-muted/40 text-sm">
          {expanded ? "▲" : "▼"}
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-0 border-t border-primary/10 mt-0 grid grid-cols-2 gap-x-6 gap-y-3 text-xs sm:grid-cols-1">
              <div>
                <div className="text-muted/60 font-mono uppercase text-xs mb-1">Location</div>
                <div className="text-light/80">{enzyme.location?.join(", ") || "—"}</div>
              </div>
              {enzyme.key_substrates?.length > 0 && (
                <div>
                  <div className="text-muted/60 font-mono uppercase text-xs mb-1">Key Substrates</div>
                  <div className="text-light/80">{enzyme.key_substrates.slice(0, 5).join(", ")}</div>
                </div>
              )}
              {enzyme.key_inhibitors?.length > 0 && (
                <div>
                  <div className="text-muted/60 font-mono uppercase text-xs mb-1">Key Inhibitors</div>
                  <div className="text-light/80">{enzyme.key_inhibitors.join(", ")}</div>
                </div>
              )}
              {enzyme.notes && (
                <div className="col-span-2 sm:col-span-1">
                  <div className="text-muted/60 font-mono uppercase text-xs mb-1">Notes</div>
                  <div className="text-muted leading-relaxed">{enzyme.notes}</div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ProbeHistoryBar({ history }) {
  if (!history || history.length < 2) return null;
  const max = history[0].size;
  return (
    <div className="mt-3">
      <div className="text-xs text-muted/50 font-mono mb-2">Probe convergence — cell size per iteration</div>
      <div className="flex items-end gap-1 h-10">
        {history.map((h) => (
          <div
            key={h.n}
            className="bg-primary/40 rounded-sm transition-all"
            style={{ width: `${Math.max(4, Math.floor(240 / history.length))}px`, height: `${Math.max(2, (h.size / max) * 40)}px` }}
            title={`iter ${h.n}: n=${h.size}, d̄=${h.meanDist.toFixed(3)}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-xs text-muted/40 mt-1 font-mono">
        <span>iter 0  n={history[0].size}</span>
        <span>iter {history[history.length - 1].n}  n={history[history.length - 1].size}</span>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Cytochrome() {
  const [uiState, setUiState] = useState(STATES.B);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef();
  const resultsRef = useRef();

  // Wake the blank screen on any keypress
  useEffect(() => {
    const handleKey = (e) => {
      if (uiState === STATES.B && !e.metaKey && !e.ctrlKey) {
        setUiState(STATES.P);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [uiState]);

  const runProbe = useCallback((q) => {
    if (!q.trim()) return;
    setLoading(true);
    setUiState(results ? STATES.AP : STATES.A);

    // Micro-task delay so the loading state renders
    setTimeout(() => {
      const intent = mapQueryToPurpose(q);
      const { stable, history } = probeIterate(cytochromeData, intent.purpose, {
        maxIter: 20,
        minSize: 2,
        rate: 0.82,
      });
      const refinements = suggestRefinements(stable);
      setResults({ intent, stable, history, refinements, query: q });
      setLoading(false);
      setUiState(STATES.A);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    }, 0);
  }, [results]);

  const handleSubmit = (e) => {
    e.preventDefault();
    runProbe(query);
  };

  const handleExampleClick = (q) => {
    setQuery(q);
    runProbe(q);
  };

  const handleRefine = (suggestion) => {
    setQuery(suggestion);
    runProbe(suggestion);
  };

  const dismiss = () => {
    setResults(null);
    setQuery("");
    setUiState(STATES.B);
  };

  return (
    <>
      <Head>
        <title>CYP450 Research Engine | Bloodhound</title>
        <meta name="description" content="Purpose-Driven Shader Virtual Machine demo: probe 57 human cytochrome P450 enzymes using S-entropy coordinates. Type a research question, get the relevant enzymes." />
      </Head>

      {/* ── Blank screen surface ──────────────────────────────────────────── */}
      <div className="min-h-screen bg-dark flex flex-col">

        {/* Minimal top bar */}
        <div className="flex items-center justify-between px-8 py-4 border-b border-primary/5 sm:px-4">
          <Link href="/" className="text-xs font-mono text-muted/50 hover:text-muted transition-colors">
            ← Bloodhound
          </Link>
          <span className="text-xs font-mono text-muted/40">CYP450 · PDSVM Demo</span>
          {results && (
            <button
              onClick={dismiss}
              className="text-xs font-mono text-muted/50 hover:text-muted transition-colors"
            >
              clear ×
            </button>
          )}
        </div>

        {/* ── Prompt surface ─────────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col items-center justify-start pt-16 px-8 sm:px-4">

          {/* Header — shown when blank or prompting */}
          <AnimatePresence>
            {(uiState === STATES.B || uiState === STATES.P) && !results && (
              <motion.div
                className="text-center mb-12 max-w-2xl"
                initial={{ opacity: 0, y: -16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.5 }}
              >
                <div className="text-xs font-mono text-primary/60 tracking-widest uppercase mb-4">
                  Purpose-Driven Shader Virtual Machine · CYP450 Demo
                </div>
                <h1 className="text-4xl font-bold text-light mb-4 md:text-3xl sm:text-2xl">
                  What are you researching?
                </h1>
                <p className="text-muted text-sm leading-relaxed">
                  Type a research question. The PDSVM probe operator maps your intent to a
                  purpose point P in S-entropy space and converges on the relevant cytochrome
                  P450 enzymes — no keyword matching, no embeddings.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Prompt input (the "blank screen" element) ─────────────────── */}
          <form
            onSubmit={handleSubmit}
            className={`w-full max-w-2xl transition-all duration-500 ${results ? "mb-8" : "mb-6"}`}
          >
            <div className="relative flex items-center">
              <span className="absolute left-4 text-muted/40 font-mono text-sm pointer-events-none">
                &gt;
              </span>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  if (uiState === STATES.B) setUiState(STATES.P);
                }}
                onFocus={() => { if (uiState === STATES.B) setUiState(STATES.P); }}
                placeholder={uiState === STATES.B ? "press any key to begin" : "research question…"}
                className="w-full bg-darkSecondary border border-primary/20 rounded-xl
                  pl-10 pr-24 py-4 text-light text-sm font-mono
                  placeholder:text-muted/30
                  focus:outline-none focus:border-primary/60 focus:ring-1 focus:ring-primary/30
                  transition-all duration-200"
              />
              <button
                type="submit"
                disabled={!query.trim() || loading}
                className="absolute right-2 px-4 py-2 rounded-lg bg-primary text-dark
                  text-xs font-bold font-mono uppercase tracking-wide
                  disabled:opacity-30 disabled:cursor-not-allowed
                  hover:bg-primary/90 transition-all duration-200"
              >
                {loading ? "…" : "probe"}
              </button>
            </div>
          </form>

          {/* Example queries — shown before first search */}
          <AnimatePresence>
            {!results && uiState !== STATES.B && (
              <motion.div
                className="w-full max-w-2xl"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="text-xs font-mono text-muted/40 mb-3 text-center">examples</div>
                <div className="flex flex-wrap gap-2 justify-center">
                  {EXAMPLE_QUERIES.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleExampleClick(q)}
                      className="text-xs px-3 py-1.5 rounded-lg border border-primary/15
                        text-muted hover:text-light hover:border-primary/35 hover:bg-surface
                        transition-all duration-200 font-mono"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Artifact surface (results) ─────────────────────────────────── */}
          <AnimatePresence>
            {results && (
              <motion.div
                ref={resultsRef}
                className="w-full max-w-5xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                {/* Intent header */}
                <div className="flex items-start justify-between mb-6 flex-wrap gap-4">
                  <div>
                    <div className="text-xs font-mono text-primary/60 uppercase tracking-widest mb-1">
                      Intent resolved
                    </div>
                    <h2 className="text-xl font-bold text-light">{results.intent.label}</h2>
                    <p className="text-muted text-sm mt-1">{results.intent.description}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs font-mono text-muted/50 mb-1">Purpose point P</div>
                    <SEntropyCoord
                      sk={results.intent.purpose[0]}
                      st={results.intent.purpose[1]}
                      se={results.intent.purpose[2]}
                    />
                    <div className="text-xs text-muted/40 mt-1 font-mono">
                      {results.stable.length} enzymes in stable slice
                    </div>
                  </div>
                </div>

                {/* Two-column layout: scatter + convergence | enzyme list */}
                <div className="grid grid-cols-[1fr_1.4fr] gap-6 lg:grid-cols-1 mb-8">

                  {/* Left: S-entropy scatter + probe history */}
                  <div>
                    <div className="bg-darkSecondary border border-primary/10 rounded-xl p-4">
                      <div className="text-xs font-mono text-muted/50 mb-3 uppercase tracking-wide">
                        S-Entropy Manifold M = [0,1]²
                      </div>
                      <SEntropyScatter
                        all={cytochromeData}
                        matched={results.stable}
                        purpose={results.intent.purpose}
                        width={440}
                        height={340}
                      />
                      <ProbeHistoryBar history={results.history} />
                    </div>

                    {/* Probe metadata */}
                    <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                      {[
                        { label: "iterations", value: results.history.length },
                        { label: "initial n", value: results.history[0]?.size ?? "—" },
                        { label: "stable n", value: results.stable.length },
                      ].map(({ label, value }) => (
                        <div key={label} className="bg-darkSecondary rounded-lg py-2 border border-primary/10">
                          <div className="text-sm font-bold font-mono text-primary">{value}</div>
                          <div className="text-xs text-muted/60 mt-0.5 font-mono">{label}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Right: enzyme cards */}
                  <div className="flex flex-col gap-2">
                    <div className="text-xs font-mono text-muted/50 uppercase tracking-wide mb-1">
                      Stable slice — click to expand
                    </div>
                    <div className="flex flex-col gap-2 max-h-[600px] overflow-y-auto pr-1">
                      {results.stable.map((enzyme, i) => (
                        <EnzymeCard
                          key={enzyme.id}
                          enzyme={enzyme}
                          rank={i}
                          isTop={i < 3}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Refinement suggestions */}
                {results.refinements?.length > 0 && (
                  <div className="border-t border-primary/10 pt-4">
                    <div className="text-xs font-mono text-muted/40 mb-2">Refine →</div>
                    <div className="flex flex-wrap gap-2">
                      {results.refinements.map((s) => (
                        <button
                          key={s}
                          onClick={() => handleRefine(s)}
                          className="text-xs px-3 py-1.5 rounded-lg border border-primary/15
                            text-muted hover:text-light hover:border-primary/35 hover:bg-surface
                            transition-all duration-200 font-mono"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Footer note */}
                <div className="mt-6 text-center text-xs text-muted/30 font-mono">
                  PDSVM probe · 57 human CYP450 isoforms · S-entropy coordinates · no server round-trip
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Blank state — breathing cursor */}
          <AnimatePresence>
            {uiState === STATES.B && (
              <motion.div
                className="mt-8 text-muted/20 font-mono text-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: [0.2, 0.7, 0.2] }}
                transition={{ repeat: Infinity, duration: 2.4 }}
              >
                _
              </motion.div>
            )}
          </AnimatePresence>

        </div>{/* end main content */}
      </div>
    </>
  );
}
