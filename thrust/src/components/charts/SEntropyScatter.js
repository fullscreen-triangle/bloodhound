import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * D3 scatter plot of the S-entropy manifold M=[0,1]² (Sk × St plane),
 * with circle radius encoding Se and colour distinguishing matched vs background.
 *
 * Props:
 *   all       – full array of 57 CYP data objects (background, grey)
 *   matched   – subset returned by the probe (highlighted, teal)
 *   purpose   – [pk, pt, pe] purpose point (rendered as a crosshair)
 *   width, height – SVG dimensions
 */
export default function SEntropyScatter({
  all = [],
  matched = [],
  purpose = null,
  width = 520,
  height = 380,
}) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current || all.length === 0) return;

    const margin = { top: 24, right: 32, bottom: 52, left: 56 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // ── Scales ─────────────────────────────────────────────────────────────
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);
    const rScale = d3.scaleSqrt().domain([0, 0.5]).range([3, 10]);

    // ── Grid ───────────────────────────────────────────────────────────────
    const gridTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    g.append("g")
      .attr("class", "grid")
      .selectAll("line.v")
      .data(gridTicks)
      .enter()
      .append("line")
      .attr("x1", (d) => xScale(d))
      .attr("x2", (d) => xScale(d))
      .attr("y1", 0)
      .attr("y2", h)
      .attr("stroke", "rgba(255,255,255,0.04)")
      .attr("stroke-width", 1);

    g.append("g")
      .selectAll("line.h")
      .data(gridTicks)
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "rgba(255,255,255,0.04)")
      .attr("stroke-width", 1);

    // ── Background points (all 57 CYPs) ───────────────────────────────────
    const matchedIds = new Set(matched.map((d) => d.id));

    g.selectAll("circle.bg")
      .data(all.filter((d) => !matchedIds.has(d.id)))
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.sk))
      .attr("cy", (d) => yScale(d.st))
      .attr("r", (d) => rScale(d.se))
      .attr("fill", "rgba(136,136,170,0.18)")
      .attr("stroke", "rgba(136,136,170,0.35)")
      .attr("stroke-width", 0.5);

    // ── Matched points ─────────────────────────────────────────────────────
    const matchedG = g
      .selectAll("g.matched")
      .data(matched)
      .enter()
      .append("g")
      .attr("class", "matched")
      .attr("transform", (d) => `translate(${xScale(d.sk)},${yScale(d.st)})`);

    // Glow halo
    matchedG
      .append("circle")
      .attr("r", (d) => rScale(d.se) + 4)
      .attr("fill", "rgba(42,157,143,0.12)")
      .attr("stroke", "none");

    // Main dot
    matchedG
      .append("circle")
      .attr("r", (d) => rScale(d.se))
      .attr("fill", "#2A9D8F")
      .attr("stroke", "#1a7a6e")
      .attr("stroke-width", 1);

    // Label (gene id) — only for matched, if not too crowded
    if (matched.length <= 12) {
      matchedG
        .append("text")
        .attr("x", (d) => rScale(d.se) + 4)
        .attr("y", 4)
        .attr("fill", "#aaeae4")
        .attr("font-size", "9px")
        .attr("font-family", "monospace")
        .text((d) => d.id);
    }

    // ── Purpose point crosshair ────────────────────────────────────────────
    if (purpose) {
      const px = xScale(purpose[0]);
      const py = yScale(purpose[1]);
      const cross = 8;

      g.append("line")
        .attr("x1", px - cross).attr("x2", px + cross)
        .attr("y1", py).attr("y2", py)
        .attr("stroke", "#E63946").attr("stroke-width", 1.5);
      g.append("line")
        .attr("x1", px).attr("x2", px)
        .attr("y1", py - cross).attr("y2", py + cross)
        .attr("stroke", "#E63946").attr("stroke-width", 1.5);
      g.append("circle")
        .attr("cx", px).attr("cy", py).attr("r", 4)
        .attr("fill", "none").attr("stroke", "#E63946").attr("stroke-width", 1.5);

      g.append("text")
        .attr("x", px + 7).attr("y", py - 7)
        .attr("fill", "#E63946").attr("font-size", "10px")
        .attr("font-family", "monospace")
        .text("P");
    }

    // ── Axes ───────────────────────────────────────────────────────────────
    const axisBottom = d3.axisBottom(xScale).ticks(5).tickSize(4);
    const axisLeft   = d3.axisLeft(yScale).ticks(5).tickSize(4);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(axisBottom)
      .call((ax) => {
        ax.selectAll("text").attr("fill", "#8888aa").attr("font-size", "10px");
        ax.selectAll(".domain").attr("stroke", "#333");
        ax.selectAll(".tick line").attr("stroke", "#333");
      });

    g.append("g")
      .call(axisLeft)
      .call((ax) => {
        ax.selectAll("text").attr("fill", "#8888aa").attr("font-size", "10px");
        ax.selectAll(".domain").attr("stroke", "#333");
        ax.selectAll(".tick line").attr("stroke", "#333");
      });

    // ── Axis labels ────────────────────────────────────────────────────────
    g.append("text")
      .attr("x", w / 2).attr("y", h + 40)
      .attr("text-anchor", "middle")
      .attr("fill", "#8888aa").attr("font-size", "11px")
      .text("Sk  (knowledge entropy — substrate breadth)");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -42)
      .attr("text-anchor", "middle")
      .attr("fill", "#8888aa").attr("font-size", "11px")
      .text("St  (temporal entropy — kinetic range)");

    // ── Legend ─────────────────────────────────────────────────────────────
    const legend = g.append("g").attr("transform", `translate(${w - 120}, 4)`);

    legend.append("circle").attr("cx", 6).attr("cy", 6).attr("r", 5)
      .attr("fill", "rgba(136,136,170,0.25)").attr("stroke", "rgba(136,136,170,0.5)").attr("stroke-width", 0.5);
    legend.append("text").attr("x", 16).attr("y", 10).attr("fill", "#8888aa").attr("font-size", "10px")
      .text("all 57 CYPs");

    legend.append("circle").attr("cx", 6).attr("cy", 22).attr("r", 5)
      .attr("fill", "#2A9D8F");
    legend.append("text").attr("x", 16).attr("y", 26).attr("fill", "#aaeae4").attr("font-size", "10px")
      .text(`matched (${matched.length})`);

    if (purpose) {
      legend.append("circle").attr("cx", 6).attr("cy", 38).attr("r", 4)
        .attr("fill", "none").attr("stroke", "#E63946").attr("stroke-width", 1.5);
      legend.append("text").attr("x", 16).attr("y", 42).attr("fill", "#E63946").attr("font-size", "10px")
        .text("purpose P");
    }

    legend.append("text").attr("x", 0).attr("y", purpose ? 58 : 42)
      .attr("fill", "#555577").attr("font-size", "9px").text("radius = Se");

  }, [all, matched, purpose, width, height]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ overflow: "visible" }}
    />
  );
}
