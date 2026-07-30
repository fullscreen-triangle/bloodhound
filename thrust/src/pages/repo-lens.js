import Head from "next/head";
import Layout from "@/components/Layout";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useCallback, useRef, useEffect } from "react";

import { analyseFederation } from "@/lib/repo-lens/analyse";
import { run as runStHurbert } from "@/lib/repo-lens/sthurbert";
import { generateStHurbert } from "@/lib/repo-lens/ai";
import { repoName, repoShortName, fileUrl } from "@/lib/repo-lens/model";

import ChiBars from "@/components/repo-lens/ChiBars";
import SalientTreemap from "@/components/repo-lens/SalientTreemap";
import FragmentGraph from "@/components/repo-lens/FragmentGraph";

/**
 * Repo Lens — a notebook/REPL over a federation of repos.
 *
 * You give a token and name repos; the browser fetches and analyses them (χ, the
 * conserved sense) in pure TypeScript. Analysis emits the first cells: a set of
 * interactive D3 charts. Then every st-Hurbert program or AI prompt you enter
 * appends a new cell below — the transcript only ever grows (a visible monotone
 * committed count, the T4/I2 invariant made tangible).
 */

// ── cell chrome ───────────────────────────────────────────────────────────────

function Cell({ n, kind, children }) {
  const badge =
    { chart: "▣ dashboard", sthurbert: "» st-Hurbert", ai: "✦ ai", error: "✗ error", note: "· note" }[kind] ||
    kind;
  const tone =
    kind === "error" ? "text-danger" : kind === "ai" ? "text-accent" : "text-primary";
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3"
    >
      <div className="shrink-0 w-14 pt-1 text-right">
        <span className="font-mono text-xs text-muted select-none">[{n}]</span>
      </div>
      <div className="flex-1 min-w-0 border-l-2 border-primary/15 pl-4 pb-6">
        <div className={`font-mono text-[11px] uppercase tracking-widest mb-2 ${tone}`}>{badge}</div>
        {children}
      </div>
    </motion.div>
  );
}

const Stat = ({ label, value, hint }) => (
  <div className="bg-surface rounded-xl p-4 border border-primary/10">
    <div className="text-[11px] font-mono text-muted uppercase tracking-widest mb-1">{label}</div>
    <div className="text-xl font-bold text-light">{value}</div>
    {hint && <div className="text-xs text-muted mt-1">{hint}</div>}
  </div>
);

const ChartCard = ({ title, children }) => (
  <div className="bg-darkSecondary/50 rounded-2xl p-5 border border-primary/10">
    <div className="text-xs font-mono text-muted uppercase tracking-widest mb-3">{title}</div>
    {children}
  </div>
);

// ── result rendering ────────────────────────────────────────────────────────

function ResultBlocks({ blocks, federation }) {
  return (
    <div className="space-y-3">
      {blocks.map((b, i) => (
        <div key={i} className="bg-surface rounded-xl p-4 border border-primary/10">
          <div className="text-xs font-mono text-primary mb-2">{b.label}</div>
          {renderBlockBody(b, federation)}
        </div>
      ))}
    </div>
  );
}

/** Turn `file:line …` result lines into clickable GitHub links when we can. */
function renderBlockBody(b, federation) {
  const rows = b.data && Array.isArray(b.data) && b.data.length && b.data[0]?.file ? b.data : null;
  if (rows) {
    return (
      <ul className="space-y-1">
        {rows.slice(0, 60).map((row, i) => {
          const repo = federation?.repos.find((r) => repoName(r) === row.repo || repoShortName(r) === row.repo);
          const href = repo ? fileUrl(repo, row.file, row.line) : null;
          const inner = (
            <span className="font-mono text-xs">
              <span className="text-muted">{row.repo ? `${row.repo} ` : ""}</span>
              <span className="text-light">{row.file}:{row.line}</span>
              <span className="text-primary"> [{row.kind}]</span> <span className="text-light/80">{row.name}</span>
            </span>
          );
          return (
            <li key={i}>
              {href ? (
                <a href={href} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-primary">
                  {inner} <span className="text-primary/60">↗</span>
                </a>
              ) : (
                inner
              )}
            </li>
          );
        })}
      </ul>
    );
  }
  return <pre className="text-sm text-muted whitespace-pre-wrap font-mono">{b.lines.join("\n")}</pre>;
}

// ── page ────────────────────────────────────────────────────────────────────

export default function RepoLens() {
  const [token, setToken] = useState("");
  const [reposInput, setReposInput] = useState("");
  const [analysing, setAnalysing] = useState(false);
  const [progress, setProgress] = useState(null);
  const [federation, setFederation] = useState(null);

  const [cells, setCells] = useState([]); // the accumulating transcript
  const seq = useRef(0);
  const endRef = useRef(null);

  // active input line
  const [mode, setMode] = useState("sthurbert"); // "sthurbert" | "ai"
  const [code, setCode] = useState("navigate * ; show chi");
  const [prompt, setPrompt] = useState("");
  const [provider, setProvider] = useState("ollama");
  const [model, setModel] = useState("llama3.2");
  const [busy, setBusy] = useState(false);

  const push = useCallback((kind, render) => {
    seq.current += 1;
    setCells((c) => [...c, { n: seq.current, kind, render }]);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [cells.length]);

  const analyse = useCallback(async () => {
    setAnalysing(true);
    const repoList = reposInput.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
    try {
      const { federation: fed, errors } = await analyseFederation(repoList, {
        token,
        maxFiles: 400,
        onProgress: (p) => setProgress(p),
      });
      setFederation(fed);
      if (fed.repos.length > 0) {
        push("chart", () => <DashboardCell federation={fed} onSelectRepo={focusRepo} />);
      }
      for (const e of errors) {
        push("error", () => <span className="text-sm text-danger font-mono">✗ {e.repo}: {e.error}</span>);
      }
      if (fed.repos.length === 0 && errors.length === 0) {
        push("note", () => <span className="text-sm text-muted">No repos resolved.</span>);
      }
    } catch (e) {
      push("error", () => <span className="text-sm text-danger font-mono">✗ {e.message}</span>);
    } finally {
      setAnalysing(false);
      setProgress(null);
    }
  }, [token, reposInput, push]);

  // clicking a repo in a chart injects a scoped query into the input line
  const focusRepo = useCallback((repo) => {
    setMode("sthurbert");
    setCode(`navigate ${repoName(repo)} ; show sense`);
  }, []);

  const runCode = useCallback(() => {
    if (!federation) return;
    const src = code.trim();
    if (!src) return;
    push("sthurbert", () => (
      <div className="space-y-3">
        <pre className="bg-dark rounded-lg px-3 py-2 text-sm text-light font-mono border border-primary/10">{src}</pre>
        <RunOutput out={runStHurbert(src, federation)} federation={federation} />
      </div>
    ));
  }, [code, federation, push]);

  const runAi = useCallback(async () => {
    if (!federation || !prompt.trim()) return;
    setBusy(true);
    const q = prompt.trim();
    try {
      const gen = await generateStHurbert(federation, { provider, model, prompt: q });
      const run = gen.ok ? runStHurbert(gen.code, federation) : null;
      push("ai", () => (
        <div className="space-y-3">
          <div className="text-sm text-light/90 italic">“{q}”</div>
          {gen.code && (
            <div>
              <div className="text-[11px] font-mono text-muted uppercase tracking-widest mb-1">
                translated → st-Hurbert (compiled ✓)
              </div>
              <pre className="bg-dark rounded-lg px-3 py-2 text-sm text-light font-mono border border-primary/10">{gen.code}</pre>
            </div>
          )}
          {!gen.ok && <div className="text-sm text-danger font-mono">ai: {gen.error}</div>}
          {run && <RunOutput out={run} federation={federation} />}
        </div>
      ));
    } catch (e) {
      push("error", () => <span className="text-sm text-danger font-mono">✗ {e.message}</span>);
    } finally {
      setBusy(false);
    }
  }, [federation, prompt, provider, model, push]);

  const onCodeKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runCode(); }
  };
  const onPromptKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runAi(); }
  };

  const ready = federation && federation.repos.length > 0;

  return (
    <>
      <Head>
        <title>Repo Lens | Bloodhound</title>
        <meta
          name="description"
          content="A notebook over a federation of repositories: give a token, fetch and compute each repo's conserved sense (χ) in the browser, then query the group with st-Hurbert code or plain language — every result appended as a cell."
        />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Repo Lens</div>
            <h1 className="section-heading">A notebook for what your repos are about</h1>
            <p className="section-subheading mb-8">
              Give a token and name some repositories. The browser fetches and analyses them —
              computing each repo&apos;s conserved sense (the character invariant χ) entirely in
              TypeScript — and drops an interactive dashboard into the notebook. From there, every
              st-Hurbert program or plain-language question you run is appended as a new cell.
            </p>

            {/* ── access ── */}
            <div className="bg-darkSecondary/60 rounded-2xl p-6 border border-primary/10 mb-8">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-1">
                <div>
                  <label className="text-xs font-mono text-muted uppercase tracking-widest">GitHub token</label>
                  <input
                    type="password"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    placeholder="ghp_… (personal-access token, repo read)"
                    className="w-full mt-1 bg-dark border border-primary/15 rounded-lg px-3 py-2 text-sm text-light font-mono focus:border-primary outline-none"
                  />
                  <div className="text-xs text-muted mt-1">Stays in your browser; sent only to api.github.com.</div>
                </div>
                <div>
                  <label className="text-xs font-mono text-muted uppercase tracking-widest">Repositories</label>
                  <textarea
                    value={reposInput}
                    onChange={(e) => setReposInput(e.target.value)}
                    placeholder={"owner/repo\nowner/other-repo  (one per line or comma-separated)"}
                    rows={3}
                    className="w-full mt-1 bg-dark border border-primary/15 rounded-lg px-3 py-2 text-sm text-light font-mono focus:border-primary outline-none resize-none"
                  />
                </div>
              </div>
              <div className="flex items-center gap-4 mt-4">
                <button
                  onClick={analyse}
                  disabled={analysing || !token || !reposInput.trim()}
                  className="bg-primary text-dark font-bold px-6 py-2.5 rounded-lg text-sm disabled:opacity-40 hover:shadow-glow transition-all"
                >
                  {analysing ? "Analysing…" : ready ? "Re-analyse" : "Analyse repos"}
                </button>
                {progress && (
                  <span className="text-xs font-mono text-muted">
                    {progress.repo}: {progress.phase} — {progress.label} ({progress.done}/{progress.total})
                  </span>
                )}
              </div>
            </div>

            {/* ── notebook transcript ── */}
            {cells.length > 0 && (
              <div className="mb-8">
                <AnimatePresence initial={false}>
                  {cells.map((c) => (
                    <Cell key={c.n} n={c.n} kind={c.kind}>
                      {c.render()}
                    </Cell>
                  ))}
                </AnimatePresence>
                <div ref={endRef} />
              </div>
            )}

            {/* ── input line (the REPL prompt) ── */}
            {ready && (
              <div className="sticky bottom-4 bg-dark/95 backdrop-blur-md rounded-2xl p-5 border border-primary/20 shadow-glow">
                <div className="flex gap-2 mb-3">
                  <button
                    onClick={() => setMode("sthurbert")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      mode === "sthurbert" ? "bg-primary text-dark" : "bg-surface text-muted hover:text-light"
                    }`}
                  >
                    st-Hurbert
                  </button>
                  <button
                    onClick={() => setMode("ai")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      mode === "ai" ? "bg-primary text-dark" : "bg-surface text-muted hover:text-light"
                    }`}
                  >
                    AI prompt
                  </button>
                  <span className="ml-auto text-[11px] font-mono text-muted self-center">⌘/Ctrl + ⏎ to run</span>
                </div>

                {mode === "sthurbert" ? (
                  <>
                    <div className="flex items-start gap-2">
                      <span className="font-mono text-primary text-sm pt-2 select-none">In [{seq.current + 1}]:</span>
                      <textarea
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        onKeyDown={onCodeKey}
                        rows={2}
                        spellCheck={false}
                        className="flex-1 bg-dark border border-primary/15 rounded-lg px-3 py-2 text-sm text-light font-mono focus:border-primary outline-none resize-none"
                      />
                      <button
                        onClick={runCode}
                        className="bg-primary text-dark font-bold px-4 py-2 rounded-lg text-sm hover:shadow-glow transition-all self-stretch"
                      >
                        Run
                      </button>
                    </div>
                    <p className="text-[11px] text-muted mt-2 font-mono">
                      navigate &lt;repo|*&gt; · slice [kind] where &lt;cond&gt; · find &quot;text&quot; · show sense|chi|salient|fragments|files|symbols
                    </p>
                  </>
                ) : (
                  <>
                    <div className="flex gap-2 mb-2">
                      <select
                        value={provider}
                        onChange={(e) => setProvider(e.target.value)}
                        className="bg-dark border border-primary/15 rounded-lg px-2 py-1.5 text-xs text-light focus:border-primary outline-none"
                      >
                        <option value="ollama">Ollama (local)</option>
                        <option value="huggingface">HuggingFace</option>
                      </select>
                      <input
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                        placeholder="model id"
                        className="flex-1 bg-dark border border-primary/15 rounded-lg px-2 py-1.5 text-xs text-light font-mono focus:border-primary outline-none"
                      />
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="font-mono text-accent text-sm pt-2 select-none">Ask [{seq.current + 1}]:</span>
                      <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={onPromptKey}
                        rows={2}
                        placeholder="which repo is most about entropy? show classes that mention models."
                        className="flex-1 bg-dark border border-primary/15 rounded-lg px-3 py-2 text-sm text-light focus:border-primary outline-none resize-none"
                      />
                      <button
                        onClick={runAi}
                        disabled={busy || !prompt.trim()}
                        className="bg-primary text-dark font-bold px-4 py-2 rounded-lg text-sm disabled:opacity-40 hover:shadow-glow transition-all self-stretch"
                      >
                        {busy ? "…" : "Ask"}
                      </button>
                    </div>
                    <p className="text-[11px] text-muted mt-2">
                      Translated to st-Hurbert and validated by the real compiler before running; a
                      rejected draft is repaired once. Keys stay server-side.
                    </p>
                  </>
                )}
              </div>
            )}
          </motion.div>
        </Layout>
      </section>
    </>
  );
}

// ── the dashboard cell (charts emitted as the first output) ───────────────────

function DashboardCell({ federation, onSelectRepo }) {
  const totalSymbols = federation.repos.reduce((s, r) => s + r.symbols.length, 0);
  const chis = federation.repos.map((r) => r.character.chi);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3 md:grid-cols-1">
        <Stat label="Repos" value={federation.repos.length} />
        <Stat label="Symbols" value={totalSymbols} hint="indexed across the group" />
        <Stat
          label="χ spread"
          value={`${Math.min(...chis).toFixed(1)}–${Math.max(...chis).toFixed(1)}`}
          hint="conserved sense range"
        />
      </div>
      <ChartCard title="Conserved sense (χ) across the federation — click a bar to focus">
        <ChiBars federation={federation} onSelectRepo={onSelectRepo} />
      </ChartCard>
      <div className="grid grid-cols-2 gap-5 md:grid-cols-1">
        <ChartCard title="Sense surface — load-bearing files (click → GitHub)">
          <SalientTreemap federation={federation} />
        </ChartCard>
        <ChartCard title="Federation constellation — radius ∝ χ, rings ∝ fragments">
          <FragmentGraph federation={federation} onSelectRepo={onSelectRepo} />
        </ChartCard>
      </div>
    </div>
  );
}

function RunOutput({ out, federation }) {
  if (out.ok) return <ResultBlocks blocks={out.result.blocks} federation={federation} />;
  return (
    <div className="text-sm text-danger font-mono">
      {out.error.stage} error: {out.error.message}
      {out.error.line ? ` (line ${out.error.line}${out.error.col ? `:${out.error.col}` : ""})` : ""}
    </div>
  );
}
