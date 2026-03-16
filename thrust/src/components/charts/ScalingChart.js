import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function ScalingChart({ width = 500, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 120, bottom: 50, left: 70 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const nSources = d3.range(1, 20);
    const centralized = nSources.map((n) => n * 65e9);
    const fedLearn = nSources.map((n) => n * 100e6);
    const fedUnderstand = nSources.map((n) => n * 350);

    const x = d3.scaleLinear().domain([1, 19]).range([0, w]);
    const y = d3.scaleLog().domain([100, 2e12]).range([h, 0]);

    // Grid
    g.selectAll(".grid")
      .data(y.ticks(6))
      .enter()
      .append("line")
      .attr("x1", 0).attr("x2", w)
      .attr("y1", (d) => y(d)).attr("y2", (d) => y(d))
      .attr("stroke", "#2A9D8F").attr("stroke-opacity", 0.06);

    const datasets = [
      { data: centralized, color: "#E63946", label: "Centralized" },
      { data: fedLearn, color: "#F4A261", label: "Fed. Learning" },
      { data: fedUnderstand, color: "#2A9D8F", label: "Fed. Understanding" },
    ];

    // Fill between centralized and understanding
    const area = d3.area()
      .x((_, i) => x(nSources[i]))
      .y0((_, i) => y(fedUnderstand[i]))
      .y1((_, i) => y(centralized[i]))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(nSources)
      .attr("d", area)
      .attr("fill", "#2A9D8F")
      .attr("opacity", 0.05);

    // Lines
    datasets.forEach((ds) => {
      const line = d3.line()
        .x((_, i) => x(nSources[i]))
        .y((d) => y(d))
        .curve(d3.curveMonotoneX);

      const path = g.append("path")
        .datum(ds.data)
        .attr("fill", "none")
        .attr("stroke", ds.color)
        .attr("stroke-width", 2.5)
        .attr("d", line);

      const totalLength = path.node().getTotalLength();
      path
        .attr("stroke-dasharray", totalLength)
        .attr("stroke-dashoffset", totalLength)
        .transition()
        .duration(1500)
        .ease(d3.easeLinear)
        .attr("stroke-dashoffset", 0);
    });

    // Legend
    const legend = g.append("g").attr("transform", `translate(${w + 10}, ${h / 2 - 40})`);
    datasets.forEach((ds, i) => {
      const ly = legend.append("g").attr("transform", `translate(0, ${i * 22})`);
      ly.append("line").attr("x1", 0).attr("x2", 16).attr("y1", 6).attr("y2", 6)
        .attr("stroke", ds.color).attr("stroke-width", 2.5);
      ly.append("text").attr("x", 22).attr("y", 10)
        .attr("fill", "#8888aa").attr("font-size", "10px").text(ds.label);
    });

    // Gap annotation
    g.append("text")
      .attr("x", x(12))
      .attr("y", y(1e6))
      .attr("text-anchor", "middle")
      .attr("fill", "#2A9D8F")
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .attr("opacity", 0.7)
      .text("10\u2078x gap");

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(6))
      .selectAll("text").attr("fill", "#8888aa");
    g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat((d) => {
      if (d >= 1e12) return `${d / 1e12}TB`;
      if (d >= 1e9) return `${d / 1e9}GB`;
      if (d >= 1e6) return `${d / 1e6}MB`;
      if (d >= 1e3) return `${d / 1e3}KB`;
      return `${d}B`;
    })).selectAll("text").attr("fill", "#8888aa");

    g.append("text").attr("x", w / 2).attr("y", h + 40)
      .attr("text-anchor", "middle").attr("fill", "#8888aa").attr("font-size", "11px")
      .text("Number of Data Sources");

    svg.selectAll(".domain").attr("stroke", "#333");
    svg.selectAll(".tick line").attr("stroke", "#333");
  }, [width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}
