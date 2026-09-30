import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import Passages from "@/components/tracker/Passages";

/**
 * Draws a chart spec sent by the tracker engine or its chat agent:
 *
 *   { type: "bar" | "pie",  title, items: [{ label, value }] }
 *   { type: "timeline",     title, items: [{ label, date, detail?, repo? }] }
 *   { type: "network",      title, nodes: [{ id, label, group }], links: [{ source, target, label? }] }
 *   { type: "table",        title, columns: [...], rows: [[...]] }
 *
 * Specs are data, never code: nothing in them is evaluated or injected as HTML.
 */

const PALETTE = ["#2A9D8F", "#F4A261", "#E76F51", "#8AB17D", "#E9C46A", "#6D8FD8", "#C77DFF", "#E63946", "#4CC9F0", "#B5838D"];

function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export function Tooltip({ tip }) {
  if (!tip) return null;
  return (
    <div
      className="pointer-events-none absolute z-20 max-w-xs rounded-lg border border-primary/30 bg-dark/95 px-3 py-2 text-[11px] text-light shadow-glow"
      style={{ left: tip.x + 12, top: tip.y + 12 }}
    >
      {tip.lines.map((l, i) => (
        <div key={i} className={i === 0 ? "font-semibold text-primary" : "text-muted"}>
          {l}
        </div>
      ))}
    </div>
  );
}

function Bars({ spec, width, onTip }) {
  const svg = useRef(null);
  useEffect(() => {
    const items = spec.items || [];
    const barH = 22;
    const left = Math.min(180, Math.max(60, d3.max(items, (d) => String(d.label).length) * 6.5));
    const h = items.length * (barH + 6) + 24;
    const x = d3.scaleLinear().domain([0, d3.max(items, (d) => d.value) || 1]).nice().range([0, width - left - 50]);
    const s = d3.select(svg.current).attr("width", width).attr("height", h);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${left},8)`);
    const row = g.selectAll("g").data(items).join("g").attr("transform", (_, i) => `translate(0,${i * (barH + 6)})`);
    row.append("text").attr("x", -8).attr("y", barH / 2).attr("dy", "0.35em").attr("text-anchor", "end")
      .attr("fill", "#c8c8dc").attr("font-size", 11).text((d) => String(d.label).slice(0, 28));
    row.append("rect").attr("height", barH).attr("rx", 4).attr("fill", (_, i) => PALETTE[i % PALETTE.length]).attr("opacity", 0.85)
      .attr("width", 0)
      .on("mousemove", (ev, d) => onTip({ x: ev.offsetX, y: ev.offsetY, lines: [String(d.label), String(d.value)] }))
      .on("mouseleave", () => onTip(null))
      .transition().duration(500).attr("width", (d) => x(d.value));
    row.append("text").attr("y", barH / 2).attr("dy", "0.35em").attr("fill", "#8888aa").attr("font-size", 10)
      .attr("x", (d) => x(d.value) + 6).text((d) => d3.format(",~r")(d.value));
  }, [spec, width, onTip]);
  return <svg ref={svg} />;
}

function Pie({ spec, width, onTip }) {
  const svg = useRef(null);
  useEffect(() => {
    const items = (spec.items || []).filter((d) => d.value > 0);
    // Legend to the right when there is room, otherwise below the ring.
    const beside = width >= 440;
    const size = Math.min(beside ? width - 200 : width, 300);
    const r = size / 2 - 10;
    const shown = items.slice(0, 14);
    const height = beside ? Math.max(size, shown.length * 18 + 16) : size + shown.length * 18 + 12;
    const s = d3.select(svg.current).attr("width", width).attr("height", height);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${size / 2},${size / 2})`);
    const arcs = d3.pie().value((d) => d.value).sort(null)(items);
    const arc = d3.arc().innerRadius(r * 0.55).outerRadius(r);
    const total = d3.sum(items, (d) => d.value);
    g.selectAll("path").data(arcs).join("path").attr("d", arc).attr("fill", (_, i) => PALETTE[i % PALETTE.length])
      .attr("stroke", "#0a0a0f").attr("stroke-width", 2)
      .on("mousemove", (ev, d) => onTip({ x: ev.offsetX, y: ev.offsetY, lines: [String(d.data.label), `${d.data.value} (${d3.format(".0%")(d.data.value / total)})`] }))
      .on("mouseleave", () => onTip(null));
    g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").attr("fill", "#f0f0f5").attr("font-size", 18).text(d3.format(",~r")(total));
    const legend = s.append("g").attr("transform", beside ? `translate(${size + 16},16)` : `translate(8,${size + 8})`);
    const li = legend.selectAll("g").data(shown).join("g").attr("transform", (_, i) => `translate(0,${i * 18})`);
    li.append("rect").attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", (_, i) => PALETTE[i % PALETTE.length]);
    li.append("text").attr("x", 16).attr("y", 9).attr("fill", "#c8c8dc").attr("font-size", 11)
      .text((d) => `${String(d.label).slice(0, 34)} · ${d.value}`);
  }, [spec, width, onTip]);
  return <svg ref={svg} />;
}

function Timeline({ spec, width, onTip, onSelect }) {
  const svg = useRef(null);
  useEffect(() => {
    const items = (spec.items || []).map((d) => ({ ...d, t: new Date(d.date) })).filter((d) => !isNaN(d.t));
    items.sort((a, b) => b.t - a.t);
    const rowH = 20;
    const left = Math.min(200, Math.max(80, d3.max(items, (d) => String(d.label).length) * 6));
    const h = items.length * rowH + 40;
    const [lo, hi] = d3.extent(items, (d) => d.t);
    const x = d3.scaleTime().domain([lo || new Date(), hi || new Date()]).nice().range([0, width - left - 20]);
    const s = d3.select(svg.current).attr("width", width).attr("height", h);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${left},10)`);
    g.append("g").attr("transform", `translate(0,${items.length * rowH + 4})`)
      .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor((width - left) / 90))))
      .call((a) => a.selectAll("text").attr("fill", "#8888aa"))
      .call((a) => a.selectAll("line,path").attr("stroke", "#333348"));
    const row = g.selectAll("g.row").data(items).join("g").attr("class", "row").attr("transform", (_, i) => `translate(0,${i * rowH})`);
    row.append("line").attr("x1", 0).attr("x2", (d) => x(d.t)).attr("y1", rowH / 2).attr("y2", rowH / 2).attr("stroke", "#2A9D8F").attr("stroke-opacity", 0.15);
    row.append("circle").attr("cx", (d) => x(d.t)).attr("cy", rowH / 2).attr("r", 5).attr("fill", "#2A9D8F")
      .style("cursor", (d) => (d.repo && onSelect ? "pointer" : "default"))
      .on("mousemove", (ev, d) => onTip({ x: ev.offsetX, y: ev.offsetY, lines: [String(d.label), d.t.toLocaleString(), d.detail ? String(d.detail) : ""].filter(Boolean) }))
      .on("mouseleave", () => onTip(null))
      .on("click", (_, d) => d.repo && onSelect && onSelect(d.repo));
    row.append("text").attr("x", -8).attr("y", rowH / 2).attr("dy", "0.35em").attr("text-anchor", "end").attr("fill", "#c8c8dc")
      .attr("font-size", 11).text((d) => String(d.label).slice(0, 32));
  }, [spec, width, onTip, onSelect]);
  return <svg ref={svg} />;
}

function Network({ spec, width, onTip, onSelect }) {
  const svg = useRef(null);
  useEffect(() => {
    const nodes = (spec.nodes || []).map((d) => ({ ...d }));
    const ids = new Set(nodes.map((n) => n.id));
    const links = (spec.links || []).filter((l) => ids.has(l.source) && ids.has(l.target)).map((d) => ({ ...d }));
    const h = Math.min(520, Math.max(280, nodes.length * 14));
    const s = d3.select(svg.current).attr("width", width).attr("height", h);
    s.selectAll("*").remove();
    const root = s.append("g");
    s.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => root.attr("transform", ev.transform)));
    const groups = Array.from(new Set(nodes.map((n) => n.group)));
    const color = (g) => (g === "repo" ? "#2A9D8F" : g === "value" ? "#F4A261" : PALETTE[groups.indexOf(g) % PALETTE.length]);
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-160))
      .force("center", d3.forceCenter(width / 2, h / 2))
      .force("collide", d3.forceCollide(16));
    const link = root.append("g").selectAll("line").data(links).join("line").attr("stroke", "#444466").attr("stroke-opacity", 0.7)
      .on("mousemove", (ev, d) => d.label && onTip({ x: ev.offsetX, y: ev.offsetY, lines: [`cue: ${d.label}`] }))
      .on("mouseleave", () => onTip(null));
    const node = root.append("g").selectAll("g").data(nodes).join("g")
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
      .on("click", (_, d) => d.group === "repo" && onSelect && onSelect(d.label))
      .on("mousemove", (ev, d) => onTip({ x: ev.offsetX, y: ev.offsetY, lines: [String(d.label), d.group] }))
      .on("mouseleave", () => onTip(null));
    node.append("circle").attr("r", (d) => (d.group === "repo" ? 8 : 5)).attr("fill", (d) => color(d.group)).attr("stroke", "#0a0a0f").attr("stroke-width", 1.5);
    node.append("text").attr("x", 10).attr("dy", "0.35em").attr("font-size", 10).attr("fill", "#c8c8dc").text((d) => String(d.label).slice(0, 24));
    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
    return () => sim.stop();
  }, [spec, width, onTip, onSelect]);
  return <svg ref={svg} />;
}

function Table({ spec }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            {(spec.columns || []).map((c, i) => (
              <th key={i} className="text-left font-medium text-muted px-2 py-1 border-b border-primary/10">{String(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(spec.rows || []).map((r, i) => (
            <tr key={i} className="odd:bg-surface/40">
              {r.map((c, j) => (
                <td key={j} className="px-2 py-1 text-light/90">{c === null || c === undefined ? "—" : String(c)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Chart({ spec, onSelect }) {
  const [ref, width] = useWidth();
  const [tip, setTip] = useState(null);
  const Kind = { bar: Bars, pie: Pie, timeline: Timeline, network: Network }[spec.type];
  return (
    <div className="rounded-xl border border-primary/15 bg-darkSecondary/60 p-4">
      {spec.title && <div className="mb-3 text-xs font-mono uppercase tracking-wider text-primary">{spec.title}</div>}
      <div ref={ref} className="relative w-full">
        {spec.type === "table" ? (
          <Table spec={spec} />
        ) : spec.type === "passages" ? (
          <Passages result={spec} repo={spec.repo} />
        ) : Kind ? (
          <Kind spec={spec} width={width} onTip={setTip} onSelect={onSelect} />
        ) : (
          <div className="text-xs text-muted">unknown chart type {String(spec.type)}</div>
        )}
        <Tooltip tip={tip} />
      </div>
    </div>
  );
}
