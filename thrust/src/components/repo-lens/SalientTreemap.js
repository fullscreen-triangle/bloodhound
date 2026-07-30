import { useMemo, useState } from "react";
import * as d3 from "d3";
import { repoName, fileUrl } from "@/lib/repo-lens/model";

/**
 * The sense surface — a treemap of the highest-weight files across the federation.
 * Each tile is a salient file; area ∝ structural weight (degree in the block graph).
 * Hover → stats; click → the file on GitHub. This is the "where the sense lives" view.
 */
export default function SalientTreemap({ federation }) {
  const [hover, setHover] = useState(null);
  const width = 640;
  const height = 380;

  const { leaves, color } = useMemo(() => {
    const children = [];
    for (const r of federation.repos) {
      const repo = repoName(r);
      for (const s of r.character.salient.slice(0, 8)) {
        children.push({
          name: s.file,
          repo,
          repoRef: r,
          weight: Math.max(0.5, s.weight),
        });
      }
    }
    const root = d3
      .hierarchy({ children })
      .sum((d) => d.weight)
      .sort((a, b) => b.value - a.value);
    d3.treemap().size([width, height]).paddingInner(2).round(true)(root);

    const repos = Array.from(new Set(children.map((c) => c.repo)));
    const color = d3
      .scaleOrdinal()
      .domain(repos)
      .range(["#2A9D8F", "#F4A261", "#E9C46A", "#8AB0AB", "#E76F51", "#6D9DC5", "#B08EA2"]);

    return { leaves: root.leaves(), color };
  }, [federation]);

  return (
    <div className="relative">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="salient file treemap">
        {leaves.map((leaf, i) => {
          const w = leaf.x1 - leaf.x0;
          const h = leaf.y1 - leaf.y0;
          const d = leaf.data;
          const active = hover === i;
          const base = leaf.parent ? d.name.split("/").pop() : d.name;
          return (
            <g
              key={i}
              transform={`translate(${leaf.x0},${leaf.y0})`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onClick={() => {
                const href = fileUrl(d.repoRef, d.name);
                if (href) window.open(href, "_blank", "noopener");
              }}
              style={{ cursor: fileUrl(d.repoRef, d.name) ? "pointer" : "default" }}
            >
              <rect
                width={w}
                height={h}
                rx={3}
                fill={color(d.repo)}
                opacity={active ? 1 : 0.78}
                stroke={active ? "#ffffff" : "transparent"}
                strokeWidth={active ? 1.5 : 0}
              />
              {w > 46 && h > 18 && (
                <text
                  x={5}
                  y={14}
                  className="fill-dark"
                  style={{ fontSize: 10, fontWeight: 600, pointerEvents: "none" }}
                >
                  {base.length * 6 > w ? base.slice(0, Math.max(1, Math.floor(w / 6))) : base}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div className="absolute top-0 right-0 bg-dark border border-primary/20 rounded-lg px-3 py-2 text-xs font-mono text-light shadow-glow max-w-xs">
          <div className="text-primary truncate">{leaves[hover].data.repo}</div>
          <div className="text-light break-all">{leaves[hover].data.name}</div>
          <div className="text-muted">weight {leaves[hover].data.weight.toFixed(1)}</div>
          <div className="text-muted">click → open on GitHub</div>
        </div>
      )}
    </div>
  );
}
