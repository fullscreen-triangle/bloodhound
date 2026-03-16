import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function CompressionChart({ width = 500, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 60, left: 70 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = [
      { source: "Genomics", available: 65e9, extracted: 227, color: "#E63946" },
      { source: "Transcriptomics", available: 50e9, extracted: 312, color: "#457B9D" },
      { source: "Proteomics", available: 120e9, extracted: 429, color: "#2A9D8F" },
    ];

    const x = d3.scaleBand().domain(data.map((d) => d.source)).range([0, w]).padding(0.3);
    const y = d3.scaleLog().domain([100, 200e9]).range([h, 0]);

    // Grid lines
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(y.ticks(6))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))
      .attr("stroke", "#2A9D8F")
      .attr("stroke-opacity", 0.08);

    // Available bars (ghost)
    g.selectAll(".bar-available")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", (d) => x(d.source))
      .attr("y", (d) => y(d.available))
      .attr("width", x.bandwidth())
      .attr("height", (d) => h - y(d.available))
      .attr("fill", (d) => d.color)
      .attr("opacity", 0.12)
      .attr("rx", 4);

    // Extracted bars (solid)
    g.selectAll(".bar-extracted")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", (d) => x(d.source))
      .attr("y", h)
      .attr("width", x.bandwidth())
      .attr("height", 0)
      .attr("fill", (d) => d.color)
      .attr("rx", 4)
      .transition()
      .duration(1200)
      .delay((_, i) => i * 200)
      .attr("y", (d) => y(d.extracted))
      .attr("height", (d) => h - y(d.extracted));

    // Reduction labels
    g.selectAll(".label")
      .data(data)
      .enter()
      .append("text")
      .attr("x", (d) => x(d.source) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.extracted) - 8)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => d.color)
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .text((d) => `${(d.available / d.extracted).toExponential(0)}x`);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("fill", "#8888aa")
      .attr("font-size", "10px");

    g.append("g")
      .call(
        d3.axisLeft(y).ticks(5).tickFormat((d) => {
          if (d >= 1e9) return `${d / 1e9}GB`;
          if (d >= 1e6) return `${d / 1e6}MB`;
          if (d >= 1e3) return `${d / 1e3}KB`;
          return `${d}B`;
        })
      )
      .selectAll("text")
      .attr("fill", "#8888aa")
      .attr("font-size", "10px");

    svg.selectAll(".domain").attr("stroke", "#333");
    svg.selectAll(".tick line").attr("stroke", "#333");
  }, [width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}
