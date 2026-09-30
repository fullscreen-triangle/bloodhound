import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { Tooltip } from "@/components/tracker/Chart";

/**
 * The federation as a force-directed knowledge graph.
 *
 * `mode = "knowledge"`: repos and the facet values okgg found, one edge per witnessed
 * triple (hover an edge for the cue word that backs it).
 * `mode = "similarity"`: repos only, joined when they carry enough of the same values
 * (Jaccard ≥ threshold); repos okgg could not tell apart are joined by a dashed edge.
 */

export const AREA_COLORS = ["#2A9D8F", "#F4A261", "#E76F51", "#8AB17D", "#E9C46A", "#6D8FD8", "#C77DFF", "#4CC9F0", "#B5838D", "#90BE6D", "#F94144", "#43AA8B"];
const FACET_COLORS = ["#F4A261", "#E9C46A", "#C77DFF", "#4CC9F0", "#F94144", "#90BE6D", "#B5838D", "#6D8FD8", "#E76F51", "#43AA8B"];

export const area = (r) => (r.key.includes("/") ? r.key.split("/")[0] : "(top level)");

const CODE = { rs: "rust", py: "python", ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", go: "go", java: "java", c: "c", cpp: "c++", h: "c", tex: "latex", jl: "julia", r: "r", kt: "kotlin", swift: "swift", cs: "c#", rb: "ruby", php: "php", ipynb: "python", lean: "lean", sol: "solidity" };
export function primaryLanguage(r) {
  for (const [ext] of r.languages || []) if (CODE[ext]) return CODE[ext];
  return "other";
}
export function hostsOf(r) {
  const hs = Array.from(new Set((r.remotes || []).map((x) => x.host).filter(Boolean))).sort();
  return hs.length ? hs.join(" + ") : "local only";
}

export function colorKey(r, by) {
  if (by === "host") return hostsOf(r);
  if (by === "language") return primaryLanguage(r);
  if (by === "recency") return r.last_commit || "";
  return area(r);
}

/** Values carried per repo key, from the triples. */
export function valuesByRepo(graph) {
  const m = new Map();
  for (const l of graph.links || []) {
    if (!m.has(l.repo)) m.set(l.repo, new Set());
    m.get(l.repo).add(l.value_id);
  }
  return m;
}

export default function FederationGraph({ graph, mode, colorBy, activeFacets, search, threshold, selected, onSelect, height = 620 }) {
  const wrap = useRef(null);
  const svgRef = useRef(null);
  const [width, setWidth] = useState(900);
  const [tip, setTip] = useState(null);

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, Math.floor(e.contentRect.width))));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  // Categorical colour scale for the chosen colouring, stable across renders.
  const repoColor = useMemo(() => {
    const repos = graph.repos || [];
    if (colorBy === "recency") {
      const ts = repos.map((r) => new Date(r.last_commit || 0).getTime()).filter((t) => t > 0);
      const s = d3.scaleSequential(d3.interpolateRgb("#34344a", "#2A9D8F")).domain(d3.extent(ts));
      return (r) => (r.last_commit ? s(new Date(r.last_commit).getTime()) : "#34344a");
    }
    const keys = Array.from(new Set(repos.map((r) => colorKey(r, colorBy)))).sort();
    const s = d3.scaleOrdinal(keys, AREA_COLORS);
    return (r) => s(colorKey(r, colorBy));
  }, [graph, colorBy]);

  const facetColor = useMemo(() => {
    const ids = (graph.facets || []).map((f) => f.id);
    return (id) => FACET_COLORS[Math.max(0, ids.indexOf(id)) % FACET_COLORS.length];
  }, [graph]);

  // Build nodes and links for the current mode and filters.
  const data = useMemo(() => {
    const repos = graph.repos || [];
    const repoNodes = repos.map((r) => ({
      id: `r:${r.key}`, kind: "repo", label: r.name, repo: r,
      // log scale: a repo with 30k files should not swallow the picture
      r: 4 + Math.min(10, Math.log2((r.files || 0) + 1) * 0.8),
    }));
    if (mode === "similarity") {
      const vals = valuesByRepo(graph);
      const links = [];
      for (let i = 0; i < repos.length; i++) {
        for (let j = i + 1; j < repos.length; j++) {
          const a = vals.get(repos[i].key) || new Set();
          const b = vals.get(repos[j].key) || new Set();
          if (!a.size || !b.size) continue;
          let inter = 0;
          for (const v of a) if (b.has(v)) inter++;
          const jac = inter / (a.size + b.size - inter);
          if (jac >= threshold) links.push({ source: `r:${repos[i].key}`, target: `r:${repos[j].key}`, weight: jac, kind: "similar" });
        }
      }
      for (const cell of graph.cells || []) {
        for (let i = 1; i < cell.length; i++) links.push({ source: `r:${cell[0]}`, target: `r:${cell[i]}`, weight: 1, kind: "cell" });
      }
      return { nodes: repoNodes, links };
    }
    const valueNodes = [];
    for (const f of graph.facets || []) {
      if (!activeFacets.has(f.id)) continue;
      for (const v of f.values || []) {
        valueNodes.push({ id: `v:${v.id}`, kind: "value", label: v.name, facet: f.id, value: v, r: 5 + Math.min(9, Math.sqrt(v.carriers || 1)) });
      }
    }
    const ids = new Set([...repoNodes, ...valueNodes].map((n) => n.id));
    const links = (graph.links || [])
      .filter((l) => activeFacets.has(l.facet))
      .map((l) => ({ source: `r:${l.repo}`, target: `v:${l.value_id}`, cue: l.cue, channel: l.channel, facet: l.facet, kind: "triple" }))
      .filter((l) => ids.has(l.source) && ids.has(l.target));
    return { nodes: [...repoNodes, ...valueNodes], links };
  }, [graph, mode, activeFacets, threshold]);

  // The simulation: rebuilt when the node set changes.
  useEffect(() => {
    const nodes = data.nodes.map((d) => ({ ...d }));
    const links = data.links.map((d) => ({ ...d }));
    const h = height;
    const svg = d3.select(svgRef.current).attr("width", width).attr("height", h);
    svg.selectAll("*").remove();
    const root = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", (ev) => root.attr("transform", ev.transform));
    svg.call(zoom).on("dblclick.zoom", null);

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id)
        .distance((l) => (l.kind === "triple" ? 150 : 50 + 120 * (1 - (l.weight || 0))))
        .strength((l) => (l.kind === "triple" ? 0.04 : 0.1 + (l.weight || 0) * 0.5)))
      // Values are hubs with dozens of carriers: strong mutual repulsion spreads them
      // out, and weak triple edges let each repo settle between the values it carries.
      .force("charge", d3.forceManyBody().strength((d) => (d.kind === "value" ? -1400 : -220)).distanceMax(900))
      .force("center", d3.forceCenter(width / 2, h / 2))
      .force("x", d3.forceX(width / 2).strength(0.03))
      .force("y", d3.forceY(h / 2).strength(0.05))
      .force("collide", d3.forceCollide((d) => d.r + (d.kind === "value" ? 14 : 6)));

    const link = root.append("g").attr("class", "links").selectAll("line").data(links).join("line")
      .attr("stroke", (d) => (d.kind === "triple" ? facetColor(d.facet) : d.kind === "cell" ? "#E63946" : "#6d6d90"))
      .attr("stroke-opacity", (d) => (d.kind === "triple" ? 0.18 : 0.25 + (d.weight || 0) * 0.6))
      .attr("stroke-width", (d) => (d.kind === "triple" ? 0.8 : 0.8 + (d.weight || 0) * 3))
      .attr("stroke-dasharray", (d) => (d.kind === "cell" ? "4 3" : null))
      .on("mousemove", (ev, d) => {
        const [x, y] = d3.pointer(ev, wrap.current);
        const lines = d.kind === "triple"
          ? [`${d.source.label} → ${d.target.label}`, `cue: “${d.cue}”`, `seen in: ${d.channel}`]
          : d.kind === "cell"
            ? [`${d.source.label} ≡ ${d.target.label}`, "okgg could not tell these apart"]
            : [`${d.source.label} ~ ${d.target.label}`, `shared values: ${d3.format(".0%")(d.weight)}`];
        setTip({ x, y, lines });
      })
      .on("mouseleave", () => setTip(null));

    const node = root.append("g").attr("class", "nodes").selectAll("g").data(nodes).join("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
      .on("click", (ev, d) => { ev.stopPropagation(); onSelect(d); })
      .on("mousemove", (ev, d) => {
        const [x, y] = d3.pointer(ev, wrap.current);
        const lines = d.kind === "repo"
          ? [d.repo.name, d.repo.key, `${hostsOf(d.repo)} · ${primaryLanguage(d.repo)}`, d.repo.last_subject ? `last: ${d.repo.last_subject}` : ""].filter(Boolean)
          : [d.label, `value of facet ${d.facet}`, `${d.value.carriers} repos · cues: ${(d.value.cues || []).slice(0, 4).join(", ")}`];
        setTip({ x, y, lines });
      })
      .on("mouseleave", () => setTip(null));

    node.filter((d) => d.kind === "repo").append("circle")
      .attr("r", (d) => d.r).attr("fill", (d) => repoColor(d.repo)).attr("stroke", "#0a0a0f").attr("stroke-width", 1.5);
    node.filter((d) => d.kind === "value").append("rect")
      .attr("width", (d) => d.r * 2).attr("height", (d) => d.r * 2).attr("x", (d) => -d.r).attr("y", (d) => -d.r)
      .attr("transform", "rotate(45)").attr("rx", 2)
      .attr("fill", (d) => facetColor(d.facet)).attr("stroke", "#0a0a0f").attr("stroke-width", 1.5);
    node.append("text")
      .attr("x", (d) => d.r + 4).attr("dy", "0.35em")
      .attr("font-size", (d) => (d.kind === "value" ? 11 : 10))
      .attr("font-weight", (d) => (d.kind === "value" ? 600 : 400))
      .attr("fill", (d) => (d.kind === "value" ? facetColor(d.facet) : "#c8c8dc"))
      .attr("paint-order", "stroke").attr("stroke", "#0a0a0f").attr("stroke-width", 3)
      .text((d) => d.label);

    svg.on("click", () => onSelect(null));

    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
    // Fit the settled layout into view once.
    sim.on("end.fit", () => {
      const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
      const [x0, x1] = d3.extent(xs), [y0, y1] = d3.extent(ys);
      if (x0 === undefined) return;
      const k = Math.min(1.6, 0.9 / Math.max((x1 - x0 + 80) / width, (y1 - y0 + 80) / h));
      svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(width / 2, h / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
    });
    return () => sim.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, width, height, repoColor, facetColor]);

  // Highlighting: search and selection dim everything not connected.
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const q = (search || "").trim().toLowerCase();
    const sel = selected ? selected.id : null;
    const near = new Set();
    if (sel) {
      near.add(sel);
      for (const l of data.links) {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        if (s === sel) near.add(t);
        if (t === sel) near.add(s);
      }
    }
    const match = (d) => {
      if (sel) return near.has(d.id);
      if (!q) return true;
      if (d.label.toLowerCase().includes(q)) return true;
      if (d.kind === "repo") return (d.repo.key || "").toLowerCase().includes(q);
      return (d.value.cues || []).some((c) => c.toLowerCase().includes(q));
    };
    svg.selectAll("g.node").attr("opacity", (d) => (match(d) ? 1 : 0.12))
      .select("circle").attr("stroke", (d) => (d.id === sel ? "#f0f0f5" : "#0a0a0f")).attr("stroke-width", (d) => (d.id === sel ? 3 : 1.5));
    svg.selectAll("g.links line").attr("opacity", (l) => {
      if (sel) return l.source.id === sel || l.target.id === sel ? 1 : 0.05;
      if (q) return match(l.source) && match(l.target) ? 1 : 0.08;
      return 1;
    });
  }, [search, selected, data]);

  return (
    <div ref={wrap} className="relative w-full overflow-hidden rounded-xl bg-dark/60">
      <svg ref={svgRef} className="block" />
      <Tooltip tip={tip} />
    </div>
  );
}
