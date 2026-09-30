import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import * as d3 from "d3";
import Chart from "@/components/tracker/Chart";
import ProposalCard from "@/components/tracker/ProposalCard";
import EngineGate, { useEngine } from "@/components/tracker/EngineGate";
import FederationGraph, { AREA_COLORS, area, colorKey, hostsOf, primaryLanguage } from "@/components/tracker/FederationGraph";
import { ago, engine, markSeen } from "@/lib/tracker/engine";

/**
 * Repo Lens → Federation: every local repo as one knowledge graph, built by okgg from
 * what each repo's README, manifests, folders and languages say, and served by the
 * local tracker engine. Click anything to focus it; the panels below break the same
 * graph down by facet, forge, area, language and activity.
 */

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-primary/10 bg-darkSecondary/60 px-4 py-3">
      <div className="text-[10px] font-mono uppercase tracking-widest text-muted">{label}</div>
      <div className="mt-1 text-xl font-semibold text-light">{value}</div>
      {hint && <div className="text-[10px] text-muted">{hint}</div>}
    </div>
  );
}

function Toggle({ options, value, onChange }) {
  return (
    <div className="flex rounded-lg bg-surface p-0.5">
      {options.map(([v, label]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${value === v ? "bg-primary text-dark" : "text-muted hover:text-light"}`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function Trajectory({ points }) {
  const ref = useRef(null);
  useEffect(() => {
    const w = ref.current.parentElement.clientWidth || 360;
    const h = 170;
    const m = { l: 34, r: 10, t: 10, b: 24 };
    const s = d3.select(ref.current).attr("width", w).attr("height", h);
    s.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, Math.max(1, points.length - 1)]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
    s.append("g").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")))
      .call((a) => a.selectAll("text").attr("fill", "#8888aa")).call((a) => a.selectAll("line,path").attr("stroke", "#333348"));
    s.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")))
      .call((a) => a.selectAll("text").attr("fill", "#8888aa")).call((a) => a.selectAll("line,path").attr("stroke", "#333348"));
    const band = d3.area().x((_, i) => x(i)).y0(y(0)).y1((d) => y(d.value || 0)).curve(d3.curveMonotoneX);
    s.append("path").datum(points).attr("d", band).attr("fill", "#2A9D8F").attr("opacity", 0.15);
    s.append("path").datum(points).attr("d", band.lineY1()).attr("fill", "none").attr("stroke", "#2A9D8F").attr("stroke-width", 2);
    s.selectAll("circle").data(points).join("circle").attr("cx", (_, i) => x(i)).attr("cy", (d) => y(d.value || 0)).attr("r", 2.5).attr("fill", "#F4A261")
      .append("title").text((d, i) => `step ${i}: V=${(d.value || 0).toFixed(3)}, ${d.facets} facets, ${d.cert} pairs told apart`);
  }, [points]);
  return <svg ref={ref} />;
}

function RepoPanel({ repo, graph, onSelectKey }) {
  const [commits, setCommits] = useState(null);
  const [proposal, setProposal] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    setCommits(null);
    setProposal(null);
    setErr(null);
    markSeen(repo.name);
  }, [repo]);
  const carried = useMemo(() => {
    const byFacet = new Map();
    for (const l of graph.links) {
      if (l.repo !== repo.key) continue;
      if (!byFacet.has(l.facet)) byFacet.set(l.facet, []);
      byFacet.get(l.facet).push(l);
    }
    return Array.from(byFacet.entries());
  }, [graph, repo]);
  const onGithub = (repo.remotes || []).some((r) => r.host === "github.com");
  const twins = (repo.okgg && repo.okgg.indiscernible_from) || [];
  const loadCommits = async () => {
    setErr(null);
    try {
      const r = await engine.call("repo_status", { repo: repo.path });
      setCommits(r.charts[0] || { type: "table", title: "commits", columns: [], rows: [] });
    } catch (e) {
      setErr(e.message);
    }
  };
  const codespace = async () => {
    setErr(null);
    try {
      setProposal((await engine.propose("codespace_open", { repo: repo.path })).proposal);
    } catch (e) {
      setErr(e.message);
    }
  };
  const maxLang = d3.max(repo.languages || [], (d) => d[1]) || 1;
  return (
    <div className="space-y-4">
      <div>
        <div className="text-lg font-semibold text-light">{repo.name}</div>
        <div className="font-mono text-[11px] text-muted break-all">{repo.key}</div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div><span className="text-muted">branch </span><span className="text-light">{repo.branch || "—"}</span></div>
        <div><span className="text-muted">commits </span><span className="text-light">{repo.commits}</span></div>
        <div><span className="text-muted">files </span><span className="text-light">{repo.files}</span></div>
        <div><span className="text-muted">last </span><span className="text-light">{ago(repo.last_commit)}</span></div>
      </div>
      {repo.last_subject && <div className="rounded bg-surface px-2 py-1.5 text-[11px] text-light/80">“{repo.last_subject}”</div>}
      <div>
        <div className="mb-1 text-[10px] font-mono uppercase tracking-wider text-muted">origins</div>
        {(repo.remotes || []).length === 0 && <div className="text-[11px] text-muted">local only</div>}
        {(repo.remotes || []).map((r) => (
          <div key={r.name} className="text-[11px]">
            <span className="text-primary">{r.name}</span> <span className="text-light/80">{r.host || r.url}</span>
            {r.account && <span className="text-muted"> · {r.account}</span>}
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] font-mono uppercase tracking-wider text-muted">files by type</div>
        {(repo.languages || []).map(([ext, n]) => (
          <div key={ext} className="flex items-center gap-2 text-[10px]">
            <span className="w-10 text-right text-muted">.{ext}</span>
            <div className="h-1.5 rounded bg-primary/70" style={{ width: `${(n / maxLang) * 60}%` }} />
            <span className="text-muted">{n}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] font-mono uppercase tracking-wider text-muted">what okgg witnessed</div>
        {carried.length === 0 && <div className="text-[11px] text-muted">no values in the active facets</div>}
        {carried.map(([facet, ls]) => (
          <div key={facet} className="mb-1 text-[11px]">
            <span className="text-muted">{facet}: </span>
            {ls.map((l) => (
              <span key={l.value_id} className="mr-2 text-accent" title={`cue “${l.cue}” in ${l.channel}`}>
                {l.value} <span className="text-muted">(“{l.cue}”)</span>
              </span>
            ))}
          </div>
        ))}
        {twins.length > 0 && (
          <div className="mt-1 text-[11px] text-danger">
            indiscernible from{" "}
            {twins.map((t) => (
              <button key={t} className="underline" onClick={() => onSelectKey(t.replace(/\.md$/, ""))}>{t.replace(/\.md$/, "")}</button>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <Link href={`/code?repo=${encodeURIComponent(repo.key)}`} className="rounded bg-primary px-3 py-1.5 text-[11px] font-semibold text-dark">
          Open code
        </Link>
        <Link href={`/tracker?repo=${encodeURIComponent(repo.name)}`} className="rounded bg-surface px-3 py-1.5 text-[11px] text-light hover:bg-darkTertiary">
          Ask the tracker
        </Link>
        <button onClick={loadCommits} className="rounded bg-surface px-3 py-1.5 text-[11px] text-light hover:bg-darkTertiary">Recent commits</button>
        {onGithub && (
          <button onClick={codespace} className="rounded bg-surface px-3 py-1.5 text-[11px] text-light hover:bg-darkTertiary">Codespace…</button>
        )}
      </div>
      {err && <div className="text-[11px] text-danger">{err}</div>}
      {proposal && <ProposalCard key={proposal.id} proposal={proposal} />}
      {commits && <Chart spec={commits} />}
    </div>
  );
}

function ValuePanel({ node, graph, onSelectKey }) {
  const f = graph.facets.find((x) => x.id === node.facet);
  const carriers = graph.links.filter((l) => l.value_id === node.value.id);
  return (
    <div className="space-y-3">
      <div>
        <div className="text-lg font-semibold text-accent">{node.label}</div>
        <div className="text-[11px] text-muted">value of facet {f ? `${f.id} (${f.property})` : node.facet} · gain {f ? f.gain : "—"}</div>
      </div>
      <div className="text-[11px] text-light/80">cues: {(node.value.cues || []).join(", ")}</div>
      <div>
        <div className="mb-1 text-[10px] font-mono uppercase tracking-wider text-muted">{carriers.length} repos carry it</div>
        {carriers.map((l) => (
          <button key={l.repo} onClick={() => onSelectKey(l.repo)} className="block text-left text-[11px] text-light hover:text-primary">
            {l.repo} <span className="text-muted">— “{l.cue}” in {l.channel}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Legend({ graph, colorBy }) {
  if (colorBy === "recency") return <div className="text-[10px] text-muted">darker = longer since the last commit</div>;
  const keys = Array.from(new Set(graph.repos.map((r) => colorKey(r, colorBy)))).sort();
  const s = d3.scaleOrdinal(keys, AREA_COLORS);
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
      {keys.map((k) => (
        <span key={k} className="flex items-center gap-1 text-[10px] text-muted">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: s(k) }} />
          {k}
        </span>
      ))}
    </div>
  );
}

function count(list, key) {
  const m = d3.rollup(list, (v) => v.length, key);
  return Array.from(m, ([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
}

function Rebuild({ onBuilt }) {
  const [generator, setGenerator] = useState("lexical");
  const [proposal, setProposal] = useState(null);
  const [err, setErr] = useState(null);
  const start = async () => {
    setErr(null);
    try {
      setProposal((await engine.propose("graph_build", { generator })).proposal);
    } catch (e) {
      setErr(e.message);
    }
  };
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Toggle options={[["lexical", "lexical (seconds)"], ["ollama", "ollama (minutes)"]]} value={generator} onChange={setGenerator} />
        <button onClick={start} className="rounded bg-surface px-3 py-1 text-[11px] text-light hover:bg-darkTertiary">Rebuild graph…</button>
      </div>
      {err && <div className="text-[11px] text-danger">{err}</div>}
      {proposal && <ProposalCard key={proposal.id} proposal={proposal} onDone={onBuilt} />}
    </div>
  );
}

export default function FederationDashboard() {
  const [status, retry] = useEngine();
  const [graph, setGraph] = useState(null);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState("knowledge");
  const [colorBy, setColorBy] = useState("area");
  const [activeFacets, setActiveFacets] = useState(new Set());
  const [search, setSearch] = useState("");
  const [threshold, setThreshold] = useState(0.4);
  const [selected, setSelected] = useState(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const g = await engine.graph();
      setGraph(g);
      setActiveFacets(new Set(g.facets.map((f) => f.id)));
      setSelected(null);
    } catch (e) {
      setErr(e.message);
    }
  }, []);

  useEffect(() => {
    if (status.state === "ready") load();
  }, [status.state, load]);

  const selectKey = useCallback(
    (key) => {
      const r = graph && graph.repos.find((x) => x.key === key);
      if (r) setSelected({ id: `r:${r.key}`, kind: "repo", label: r.name, repo: r });
    },
    [graph]
  );

  const charts = useMemo(() => {
    if (!graph) return null;
    const values = graph.facets
      .flatMap((f) => f.values.map((v) => ({ label: `${v.name} · ${f.id}`, value: v.carriers || 0 })))
      .sort((a, b) => b.value - a.value)
      .slice(0, 16);
    const timeline = [...graph.repos]
      .filter((r) => r.last_commit)
      .sort((a, b) => (b.last_commit > a.last_commit ? 1 : -1))
      .slice(0, 25)
      .map((r) => ({ label: r.name, date: r.last_commit, detail: r.last_subject, repo: r.key }));
    return {
      values: { type: "bar", title: "Facet values — repos carrying each", items: values },
      hosts: { type: "pie", title: "Where the repos live", items: count(graph.repos, hostsOf) },
      areas: { type: "bar", title: "Repos per area", items: count(graph.repos, area) },
      languages: { type: "pie", title: "Primary language", items: count(graph.repos, primaryLanguage) },
      timeline: { type: "timeline", title: "Last commit — 25 most recent (click to focus)", items: timeline },
    };
  }, [graph]);

  return (
    <EngineGate status={status} retry={retry}>
      {err && (
        <div className="rounded-xl border border-danger/30 bg-danger/5 p-4 text-sm text-danger">
          {err}
          <div className="mt-3"><Rebuild onBuilt={load} /></div>
        </div>
      )}
      {graph && (
        <div className="space-y-6">
          <div className="grid grid-cols-6 gap-3 xl:grid-cols-3 sm:grid-cols-2">
            <Stat label="Repos" value={graph.repos.length} hint={`${graph.hosts.length} forges`} />
            <Stat label="Facets" value={graph.facets.length} hint={`${graph.facets.reduce((s, f) => s + f.values.length, 0)} values`} />
            <Stat label="Triples" value={graph.links.length} hint="each backed by a cue" />
            <Stat label="Value V" value={graph.okgg.value != null ? graph.okgg.value.toFixed(3) : "—"} hint="pairs told apart" />
            <Stat label="Indiscernible" value={graph.okgg.indiscernible ?? "—"} hint={`${graph.okgg.open ?? 0} still open`} />
            <Stat label="Generator" value={graph.generator} hint={new Date(graph.generated_at * 1000).toLocaleDateString()} />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Toggle options={[["knowledge", "Knowledge graph"], ["similarity", "Repo similarity"]]} value={mode} onChange={setMode} />
            <Toggle options={[["area", "area"], ["host", "forge"], ["language", "language"], ["recency", "recency"]]} value={colorBy} onChange={setColorBy} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="search repos, values, cues…"
              className="w-56 rounded-lg border border-primary/15 bg-surface px-3 py-1.5 text-xs text-light placeholder:text-muted focus:border-primary/50 focus:outline-none"
            />
            {mode === "similarity" && (
              <label className="flex items-center gap-2 text-[11px] text-muted">
                shared ≥ {d3.format(".0%")(threshold)}
                <input type="range" min="0.1" max="0.9" step="0.05" value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
              </label>
            )}
            <div className="ml-auto"><Rebuild onBuilt={load} /></div>
          </div>

          {mode === "knowledge" && (
            <div className="flex flex-wrap gap-2">
              {graph.facets.map((f) => {
                const on = activeFacets.has(f.id);
                return (
                  <button
                    key={f.id}
                    onClick={() => {
                      const next = new Set(activeFacets);
                      on ? next.delete(f.id) : next.add(f.id);
                      setActiveFacets(next);
                    }}
                    className={`rounded-full border px-3 py-1 text-[11px] transition-colors ${on ? "border-accent/60 bg-accent/10 text-accent" : "border-primary/10 text-muted"}`}
                    title={`gain ${f.gain}`}
                  >
                    {f.id}: {f.values.map((v) => v.name).join(" / ")}
                  </button>
                );
              })}
            </div>
          )}

          <div className="grid grid-cols-4 gap-5 lg:grid-cols-1">
            <div className="col-span-3 lg:col-span-1 space-y-2">
              <FederationGraph
                graph={graph}
                mode={mode}
                colorBy={colorBy}
                activeFacets={activeFacets}
                search={search}
                threshold={threshold}
                selected={selected}
                onSelect={setSelected}
              />
              <Legend graph={graph} colorBy={colorBy} />
            </div>
            <div className="rounded-xl border border-primary/10 bg-darkSecondary/60 p-4 max-h-[680px] overflow-y-auto">
              {!selected && (
                <div className="text-xs text-muted space-y-2">
                  <p>Click a repo (circle) or a value (diamond) to focus it. Drag to move, scroll to zoom.</p>
                  <p>Edges are okgg triples: a repo carries a value, witnessed by a cue word found in what the repo says about itself. Hover one to see the cue.</p>
                </div>
              )}
              {selected && selected.kind === "repo" && <RepoPanel repo={selected.repo} graph={graph} onSelectKey={selectKey} />}
              {selected && selected.kind === "value" && <ValuePanel node={selected} graph={graph} onSelectKey={selectKey} />}
            </div>
          </div>

          {charts && (
            <div className="grid grid-cols-2 gap-5 lg:grid-cols-1">
              <Chart spec={charts.values} />
              <div className="grid grid-cols-2 gap-5 md:grid-cols-1">
                <Chart spec={charts.hosts} />
                <Chart spec={charts.languages} />
              </div>
              <Chart spec={charts.timeline} onSelect={selectKey} />
              <div className="space-y-5">
                <Chart spec={charts.areas} />
                <div className="rounded-xl border border-primary/15 bg-darkSecondary/60 p-4">
                  <div className="mb-3 text-xs font-mono uppercase tracking-wider text-primary">
                    okgg trajectory — value V per step ({graph.okgg.stop})
                  </div>
                  <div className="w-full"><Trajectory points={graph.okgg.trajectory || []} /></div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </EngineGate>
  );
}
