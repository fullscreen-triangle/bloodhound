import React, { useEffect, useRef } from 'react';
import ForceGraph3D from '3d-force-graph';

export const KnowledgeGraph = ({ data, onNodeClick }) => {
  const graphRef = useRef(null);
  
  useEffect(() => {
    if (!graphRef.current) return;
    
    const Graph = ForceGraph3D()
      (graphRef.current)
      .graphData(data)
      .nodeLabel('label')
      .nodeColor(node => getNodeColor(node.type))
      .linkLabel('label')
      .onNodeClick(onNodeClick);
      
    return () => {
      Graph.dispose();
    };
  }, [data, onNodeClick]);
  
  return (
    <div 
      ref={graphRef} 
      style={{ width: '100%', height: '600px' }} 
      className="knowledge-graph-container"
    />
  );
};

const getNodeColor = (type) => {
  const colors = {
    experiment: '#ff7f0e',
    result: '#1f77b4',
    metadata: '#2ca02c'
  };
  return colors[type] || '#7f7f7f';
}; 