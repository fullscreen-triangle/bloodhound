import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';

const SearchBar = ({ onResultsChange }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const debounceTimer = useRef(null);
  
  useEffect(() => {
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }
    
    // Debounce search suggestions
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    
    debounceTimer.current = setTimeout(async () => {
      try {
        setIsLoading(true);
        const suggestionsData = await api.getSuggestions(query);
        setSuggestions(suggestionsData);
      } catch (error) {
        console.error('Error fetching suggestions:', error);
      } finally {
        setIsLoading(false);
      }
    }, 300);
    
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [query]);
  
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    try {
      setIsLoading(true);
      
      // Search across different data sources
      const results = await api.search({
        query,
        sources: ['experiments', 'knowledge_graph', 'embeddings'],
        filters: {
          // Add any active filters here
        }
      });
      
      onResultsChange(results);
      
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="search-container">
      <form onSubmit={handleSearch} className="search-form">
        <input 
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search experiments, methods, or results..."
          className="search-input"
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="search-button"
          disabled={isLoading}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </form>
      
      {suggestions.length > 0 && (
        <div className="suggestions-dropdown">
          {suggestions.map(suggestion => (
            <div 
              key={suggestion.id}
              className="suggestion-item"
              onClick={() => {
                setQuery(suggestion.text);
                handleSearch({ preventDefault: () => {} });
              }}
            >
              <span className="suggestion-text">{suggestion.text}</span>
              <span className="suggestion-type">{suggestion.type}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
