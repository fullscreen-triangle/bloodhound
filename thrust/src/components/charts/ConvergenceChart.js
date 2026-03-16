import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function ConvergenceChart({ width = 500, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Generate convergence data
    const data = [];
    let temp = 0.45;
    for (let i = 0; i < 30; i++) {
      data.push({ step: i, temperature: temp, phase: temp > 0.5 ? "gas" : temp > 0.2 ? "liquid" : "crystal" });
      temp = temp * 0.92 + (Math.random() - 0.5) * 0.01;
      temp = Math.max(0.05, temp);
    }

    const x = d3.scaleLinear().domain([0, 29]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 0.6]).range([h, 0]);

    // Phase regions
    const phases = [
      { y0: 0, y1: 0.2, color: "#2A9D8F", label: "Crystal" },
      { y0: 0.2, y1: 0.5, color: "#F4A261", label: "Liquid" },
      { y0: 0.5, y1: 0.6, color: "#E63946", label: "Gas" },
    ];

    phases.forEach((p) => {
      g.append("rect")
        .attr("x", 0)
        .attr("y", y(p.y1))
        .attr("width", w)
        .attr("height", y(p.y0) - y(p.y1))
        .attr("fill", p.color)
        .attr("opacity", 0.06);

      g.append("text")
        .attr("x", w - 5)
        .attr("y", y((p.y0 + p.y1) / 2))
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("fill", p.color)
        .attr("font-size", "10px")
        .attr("opacity", 0.6)
        .text(p.label);
    });

    // Line
    const line = d3.line().x((d) => x(d.step)).y((d) => y(d.temperature)).curve(d3.curveMonotoneX);

    const path = g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#2A9D8F")
      .attr("stroke-width", 2.5)
      .attr("d", line);

    // Animate the line drawing
    const totalLength = path.node().getTotalLength();
    path
      .attr("stroke-dasharray", totalLength)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(2000)
      .ease(d3.easeLinear)
      .attr("stroke-dashoffset", 0);

    // Data points
    g.selectAll(".dot")
      .data(data.filter((_, i) => i % 3 === 0))
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.step))
      .attr("cy", (d) => y(d.temperature))
      .attr("r", 4)
      .attr("fill", (d) => d.phase === "gas" ? "#E63946" : d.phase === "liquid" ? "#F4A261" : "#2A9D8F")
      .attr("stroke", "#0a0a0f")
      .attr("stroke-width", 2)
      .attr("opacity", 0)
      .transition()
      .delay((_, i) => i * 200 + 500)
      .duration(300)
      .attr("opacity", 1);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll("text")
      .attr("fill", "#8888aa");

    g.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .selectAll("text")
      .attr("fill", "#8888aa");

    // Axis labels
    g.append("text")
      .attr("x", w / 2)
      .attr("y", h + 40)
      .attr("text-anchor", "middle")
      .attr("fill", "#8888aa")
      .attr("font-size", "11px")
      .text("Analysis Step");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -h / 2)
      .attr("y", -45)
      .attr("text-anchor", "middle")
      .attr("fill", "#8888aa")
      .attr("font-size", "11px")
      .text("Temperature");

    svg.selectAll(".domain").attr("stroke", "#333");
    svg.selectAll(".tick line").attr("stroke", "#333");
  }, [width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}
