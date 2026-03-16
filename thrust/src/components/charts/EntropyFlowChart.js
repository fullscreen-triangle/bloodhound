import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function EntropyFlowChart({ width = 500, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 50, left: 60 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const stages = ["Genomics", "Cardiac", "Proteomics", "Compose 1", "Compose 2"];
    const sk = [0.30, 0.35, 0.40, 0.27, 0.29];
    const st = [0.10, 0.20, 0.15, 0.20, 0.21];
    const se = [0.60, 0.45, 0.45, 0.53, 0.50];

    const x = d3.scaleBand().domain(stages).range([0, w]).padding(0.2);
    const y = d3.scaleLinear().domain([0, 1.1]).range([h, 0]);

    // Stacked bars
    const layers = [
      { key: "Sk", data: sk, color: "#E63946" },
      { key: "St", data: st, color: "#457B9D" },
      { key: "Se", data: se, color: "#2A9D8F" },
    ];

    stages.forEach((stage, i) => {
      let cumY = 0;
      layers.forEach((layer) => {
        g.append("rect")
          .attr("x", x(stage))
          .attr("y", y(cumY + layer.data[i]))
          .attr("width", x.bandwidth())
          .attr("height", y(cumY) - y(cumY + layer.data[i]))
          .attr("fill", layer.color)
          .attr("opacity", 0.8)
          .attr("rx", 2);
        cumY += layer.data[i];
      });

      // Total line point
      g.append("circle")
        .attr("cx", x(stage) + x.bandwidth() / 2)
        .attr("cy", y(cumY))
        .attr("r", 4)
        .attr("fill", "#fff")
        .attr("stroke", "#0a0a0f")
        .attr("stroke-width", 2);
    });

    // Total line
    const totalLine = d3.line()
      .x((_, i) => x(stages[i]) + x.bandwidth() / 2)
      .y((d) => y(d))
      .curve(d3.curveMonotoneX);

    const totals = stages.map((_, i) => sk[i] + st[i] + se[i]);
    g.append("path")
      .datum(totals)
      .attr("fill", "none")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "4,4")
      .attr("d", totalLine);

    // Legend
    const legend = g.append("g").attr("transform", `translate(${w - 120}, 0)`);
    layers.forEach((layer, i) => {
      const ly = legend.append("g").attr("transform", `translate(0, ${i * 18})`);
      ly.append("rect").attr("width", 12).attr("height", 12).attr("fill", layer.color).attr("rx", 2);
      ly.append("text").attr("x", 18).attr("y", 10).attr("fill", "#8888aa").attr("font-size", "11px").text(layer.key);
    });

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("fill", "#8888aa")
      .attr("font-size", "9px")
      .attr("transform", "rotate(-20)")
      .attr("text-anchor", "end");

    g.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .selectAll("text")
      .attr("fill", "#8888aa");

    svg.selectAll(".domain").attr("stroke", "#333");
    svg.selectAll(".tick line").attr("stroke", "#333");
  }, [width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}
