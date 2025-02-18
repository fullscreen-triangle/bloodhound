import React from 'react';
import Plot from 'react-plotly.js';

export const ExperimentPlot = ({ plotData, config }) => {
  return (
    <Plot
      data={plotData.data}
      layout={plotData.layout}
      config={{
        responsive: true,
        ...config
      }}
      style={{ width: '100%', height: '400px' }}
      className="experiment-plot"
    />
  );
}; 