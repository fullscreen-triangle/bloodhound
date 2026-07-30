import { useMemo, useRef, useState, useEffect } from "react";
import * as d3 from "d3";
import { repoName, repoShortName } from "@/lib/repo-lens/model";

/**
 * The federation as a constellation: one node per repo, radius ∝ χ (conserved
 * sense-mass), a faint ring showing fragmentation (how far the repo is from a
 * single connected body of work). Hover → stats + the cheapest cut; click → focus.
 * A d3-force layout settles the nodes; React renders them.
 */
export default function FragmentGraph({ federation, onSelectRepo }) {
  const [hover, setHover] = useState(null);
  const [nodes, setNodes] = useState([]);
  const width = 640;
  const height = 380;

  const seed = useMemo(
    () =>
      federation.repos.map((r, i) => ({
        id: repoName(r),
        repo: r,
        chi: r.character.chi,
        fragments: r.character.fragments,
        idx: i,
      })),
    [federation]
  );

  const rScale = useMemo(() => {
    const max = d3.max(seed, (d) => d.chi) || 1;
    return d3.scaleSqrt().domain([0, max]).range([10, 46]);
  }, [seed]);

  useEffect(() => {
    const sim = d3
      .forceSimulation(seed)
      .force("charge", d3.forceManyBody().strength(-140))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collide",
        d3.forceCollide().radius((d) => rScale(d.chi) + 14)
      )
      .stop();
    for (let i = 0; i < 220; i++) sim.tick();
    setNodes(seed.map((d) => ({ ...d, x: d.x, y: d.y })));
    return () => sim.stop();
  }, [seed, rScale]);

  return (
    <div className="relative">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="federation constellation">
        {nodes.map((n, i) => {
          const r = rScale(n.chi);
          const active = hover === i;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelectRepo?.(n.repo)}
              style={{ cursor: "pointer" }}
            >
              {n.fragments > 1 &&
                Array.from({ length: Math.min(n.fragments - 1, 4) }).map((_, k) => (
                  <circle key={k} r={r + 5 + k * 4} fill="none" stroke="#E63946" strokeWidth={0.6} opacity={0.35} />
                ))}
              <circle
                r={r}
                fill={active ? "#F4A261" : "#2A9D8F"}
                opacity={active ? 1 : 0.85}
                stroke={active ? "#ffffff" : "#2A9D8F"}
                strokeWidth={active ? 2 : 0}
              />
              <text
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-dark"
                style={{ fontSize: 10, fontWeight: 700, pointerEvents: "none" }}
              >
                {n.chi.toFixed(0)}
              </text>
              <text
                y={r + 12}
                textAnchor="middle"
                className="fill-muted"
                style={{ fontSize: 9, fontFamily: "monospace", pointerEvents: "none" }}
              >
                {repoShortName(n.repo).length > 16
                  ? repoShortName(n.repo).slice(0, 15) + "…"
                  : repoShortName(n.repo)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover != null && nodes[hover] && (
        <div className="absolute top-0 left-0 bg-dark border border-primary/20 rounded-lg px-3 py-2 text-xs font-mono text-light shadow-glow max-w-xs">
          <div className="text-primary">{nodes[hover].id}</div>
          <div>χ = {nodes[hover].chi.toFixed(3)}</div>
          <div className="text-muted">
            {nodes[hover].repo.character.coreBlocks}/{nodes[hover].repo.character.blocks} blocks ·{" "}
            {nodes[hover].fragments} fragment(s)
          </div>
          {nodes[hover].repo.character.cutSide?.length > 0 && (
            <div className="text-danger/80 mt-1">
              cheapest cut: {nodes[hover].repo.character.cutSide.slice(0, 3).join(", ")}
              {nodes[hover].repo.character.cutSide.length > 3 ? " …" : ""}
            </div>
          )}
          <div className="text-muted mt-1">click → focus</div>
        </div>
      )}
    </div>
  );
}
