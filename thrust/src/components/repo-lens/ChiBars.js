import { useMemo, useState } from "react";
import * as d3 from "d3";
import { repoName } from "@/lib/repo-lens/model";

/**
 * χ across the federation — a horizontal bar per repo. Hover a bar for stats;
 * click to jump to the repo on GitHub. Interactive, D3-scaled, React-rendered.
 */
export default function ChiBars({ federation, onSelectRepo }) {
  const [hover, setHover] = useState(null);
  const width = 640;
  const rowH = 34;
  const padL = 160;
  const padR = 60;
  const height = federation.repos.length * rowH + 20;

  const { rows, x } = useMemo(() => {
    const data = federation.repos.map((r) => ({
      repo: r,
      name: repoName(r),
      chi: r.character.chi,
    }));
    const maxChi = d3.max(data, (d) => d.chi) || 1;
    const x = d3.scaleLinear().domain([0, maxChi]).range([0, width - padL - padR]);
    return { rows: data, x };
  }, [federation]);

  return (
    <div className="relative">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="chi across repositories">
        {rows.map((d, i) => {
          const y = i * rowH + 10;
          const w = Math.max(2, x(d.chi));
          const active = hover === i;
          return (
            <g
              key={d.name}
              transform={`translate(0,${y})`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelectRepo?.(d.repo)}
              style={{ cursor: "pointer" }}
            >
              <text x={padL - 10} y={rowH / 2} textAnchor="end" dominantBaseline="middle"
                className="fill-muted" style={{ fontSize: 11, fontFamily: "monospace" }}>
                {d.name.length > 22 ? "…" + d.name.slice(-21) : d.name}
              </text>
              <rect x={padL} y={6} width={x.range()[1]} height={rowH - 14} rx={4}
                className="fill-primary/5" />
              <rect x={padL} y={6} width={w} height={rowH - 14} rx={4}
                fill={active ? "#F4A261" : "#2A9D8F"} opacity={active ? 1 : 0.85} />
              <text x={padL + w + 8} y={rowH / 2} dominantBaseline="middle"
                className="fill-light" style={{ fontSize: 12, fontWeight: 700 }}>
                {d.chi.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div className="absolute top-0 right-0 bg-dark border border-primary/20 rounded-lg px-3 py-2 text-xs font-mono text-light shadow-glow">
          <div className="text-primary">{rows[hover].name}</div>
          <div>χ = {rows[hover].chi.toFixed(3)}</div>
          <div className="text-muted">
            {rows[hover].repo.character.coreBlocks}/{rows[hover].repo.character.blocks} blocks ·{" "}
            {rows[hover].repo.character.fragments} fragment(s)
          </div>
          <div className="text-muted">click → open on GitHub / focus</div>
        </div>
      )}
    </div>
  );
}
