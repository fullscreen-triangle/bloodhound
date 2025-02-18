import React, { useState } from 'react';
import { ExperimentPlot } from './ExperimentPlot';
import { KnowledgeGraph } from './KnowledgeGraph';

const Results = ({ results, onResultSelect }) => {
  const [activeTab, setActiveTab] = useState('all');
  const [selectedResult, setSelectedResult] = useState(null);
  
  const filterResults = (results) => {
    if (activeTab === 'all') return results;
    return results.filter(result => result.type === activeTab);
  };
  
  const handleResultClick = async (result) => {
    setSelectedResult(result);
    if (onResultSelect) {
      onResultSelect(result);
    }
  };
  
  const renderResultContent = (result) => {
    switch (result.type) {
      case 'experiment':
        return (
          <div className="experiment-result">
            <h3>{result.title}</h3>
            <p>{result.description}</p>
            {result.plots && result.plots.map(plot => (
              <ExperimentPlot key={plot.id} plotData={plot} />
            ))}
          </div>
        );
        
      case 'knowledge_graph':
        return (
          <div className="graph-result">
            <KnowledgeGraph data={result.graphData} />
          </div>
        );
        
      case 'embedding':
        return (
          <div className="embedding-result">
            <h3>Similar Results</h3>
            <ul className="similarity-list">
              {result.similar_items.map(item => (
                <li key={item.id} className="similarity-item">
                  <span className="similarity-score">
                    {(item.score * 100).toFixed(1)}% match
                  </span>
                  <span className="similarity-title">{item.title}</span>
                </li>
              ))}
            </ul>
          </div>
        );
        
      default:
        return (
          <div className="text-result">
            <h3>{result.title}</h3>
            <p>{result.content}</p>
          </div>
        );
    }
  };
  
  return (
    <div className="results-container">
      <div className="results-tabs">
        <button 
          className={`tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          All Results
        </button>
        <button 
          className={`tab ${activeTab === 'experiment' ? 'active' : ''}`}
          onClick={() => setActiveTab('experiment')}
        >
          Experiments
        </button>
        <button 
          className={`tab ${activeTab === 'knowledge_graph' ? 'active' : ''}`}
          onClick={() => setActiveTab('knowledge_graph')}
        >
          Knowledge Graphs
        </button>
        <button 
          className={`tab ${activeTab === 'embedding' ? 'active' : ''}`}
          onClick={() => setActiveTab('embedding')}
        >
          Similar Items
        </button>
      </div>
      
      <div className="results-list">
        {filterResults(results).map(result => (
          <div 
            key={result.id}
            className={`result-item ${selectedResult?.id === result.id ? 'selected' : ''}`}
            onClick={() => handleResultClick(result)}
          >
            {renderResultContent(result)}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Results;
