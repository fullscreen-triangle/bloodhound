import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = {
  async getKnowledgeGraph(experimentType) {
    const response = await axios.get(`${API_BASE_URL}/knowledge/${experimentType}`);
    return response.data;
  },
  
  async queryKnowledgeGraph(query, experimentType) {
    const response = await axios.post(`${API_BASE_URL}/knowledge/query`, {
      experiment_type: experimentType,
      sparql_query: query
    });
    return response.data.results;
  },
  
  async generatePlot(plotData) {
    const response = await axios.post(`${API_BASE_URL}/plot/generate`, plotData);
    return response.data;
  },
  
  async getPlot(plotId) {
    const response = await axios.get(`${API_BASE_URL}/plot/${plotId}`);
    return response.data;
  },
  
  async sendChatMessage(data) {
    const response = await axios.post(`${API_BASE_URL}/chat/message`, data);
    return response.data;
  }
}; 