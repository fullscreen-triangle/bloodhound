import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function ParadigmChart({ width = 500, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 30, right: 30, bottom: 50, left: 80 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const paradigms = [
      { name: "Centralized", transfer: 235e9, color: "#E63946", complexity: "O(|D|)" },
      { name: "Federated\nLearning", transfer: 300e6, color: "#F4A261", complexity: "O(H(D))" },
      { name: "Federated\nUnderstanding", transfer: 968, color: "#2A9D8F", complexity: "O(I(D;A_Q))" },
    ];

    const x = d3.scaleBand().domain(paradigms.map((d) => d.name)).range([0, w]).padding(0.4);
    const y = d3.scaleLog().domain([100, 500e9]).range([h, 0]);

    // Grid
    g.selectAll(".grid-line")
      .data(y.ticks(6))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))
      .attr("stroke", "#2A9D8F")
      .attr("stroke-opacity", 0.06);

    // Bars with animation
    g.selectAll(".bar")
      .data(paradigms)
      .enter()
      .append("rect")
      .attr("x", (d) => x(d.name))
      .attr("y", h)
      .attr("width", x.bandwidth())
      .attr("height", 0)
      .attr("fill", (d) => d.color)
      .attr("rx", 6)
      .transition()
      .duration(1500)
      .delay((_, i) => i * 300)
      .attr("y", (d) => y(d.transfer))
      .attr("height", (d) => h - y(d.transfer));

    // Glow effect for understanding bar
    const defs = svg.append("defs");
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Value labels
    g.selectAll(".value-label")
      .data(paradigms)
      .enter()
      .append("text")
      .attr("x", (d) => x(d.name) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.transfer) - 12)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => d.color)
      .attr("font-size", "12px")
      .attr("font-weight", "bold")
      .text((d) => {
        if (d.transfer >= 1e9) return `${(d.transfer / 1e9).toFixed(1)} GB`;
        if (d.transfer >= 1e6) return `${(d.transfer / 1e6).toFixed(1)} MB`;
        return `${d.transfer} B`;
      });

    // Complexity labels
    g.selectAll(".complexity")
      .data(paradigms)
      .enter()
      .append("text")
      .attr("x", (d) => x(d.name) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.transfer) - 28)
      .attr("text-anchor", "middle")
      .attr("fill", "#8888aa")
      .attr("font-size", "10px")
      .text((d) => d.complexity);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("fill", "#8888aa")
      .attr("font-size", "10px")
      .style("white-space", "pre");

    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickFormat((d) => {
        if (d >= 1e9) return `${d / 1e9}GB`;
        if (d >= 1e6) return `${d / 1e6}MB`;
        if (d >= 1e3) return `${d / 1e3}KB`;
        return `${d}B`;
      }))
      .selectAll("text")
      .attr("fill", "#8888aa")
      .attr("font-size", "10px");

    svg.selectAll(".domain").attr("stroke", "#333");
    svg.selectAll(".tick line").attr("stroke", "#333");
  }, [width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}
