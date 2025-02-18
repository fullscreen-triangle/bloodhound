from typing import Dict, Any, List, Optional
import redis
from pymongo import MongoClient
import h5py
import networkx as nx
from distributed import Client
import dask.dataframe as dd
import pandas as pd
import json
from pathlib import Path
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef
import plotly.graph_objects as go
import plotly.express as px
import logging
import time

class DistributedDBManager:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.base_path = Path("./data")
        
        # Initialize storage paths
        self.experiment_path = self.base_path / "experiments"
        self.knowledge_path = self.base_path / "knowledge_graphs"
        self.plot_cache = self.base_path / "plot_cache"
        
        for path in [self.experiment_path, self.knowledge_path, self.plot_cache]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connections
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['science_db']
        
        # Initialize Dask client
        if self.resource_manager.compute_env == "HPC":
            self.dask_client = Client(scheduler_file='scheduler.json')
        else:
            self.dask_client = Client(n_workers=self.resource_manager._calculate_workers())
    
    async def store_experiment(self, data: Dict[str, Any]):
        """Store experiment data across distributed storage"""
        exp_id = data['metadata']['id']
        exp_type = data['metadata']['type']
        
        try:
            # Store metadata in MongoDB
            self.db.experiments.insert_one(data['metadata'])
            
            # Store large datasets in HDF5
            with h5py.File(self.experiment_path / f"{exp_id}.h5", 'w') as f:
                # Store embeddings
                if 'embeddings' in data:
                    f.create_dataset('embeddings', data=data['embeddings'])
                
                # Store experiment-specific results
                if 'results' in data:
                    results_group = f.create_group('results')
                    self._store_results_in_hdf5(results_group, data['results'])
                
                # Store plots data
                if 'plots' in data:
                    plots_group = f.create_group('plots')
                    for plot_name, plot_data in data['plots'].items():
                        self._store_plot_data(plots_group, plot_name, plot_data)
            
            # Update knowledge graph
            await self._update_knowledge_graph(exp_id, data)
            
            # Cache frequently accessed data in Redis
            self._cache_experiment_data(exp_id, data)
            
            return exp_id
            
        except Exception as e:
            logging.error(f"Error storing experiment {exp_id}: {str(e)}")
            raise
    
    async def retrieve_experiment(self, 
                                experiment_id: str,
                                include_plots: bool = True) -> Dict[str, Any]:
        """Retrieve experiment data from distributed storage"""
        try:
            # Get metadata from MongoDB
            metadata = self.db.experiments.find_one({'id': experiment_id})
            
            # Check Redis cache first
            cached_data = self._get_cached_data(experiment_id)
            if cached_data:
                return cached_data
            
            # Load from HDF5
            result = {'metadata': metadata}
            with h5py.File(self.experiment_path / f"{experiment_id}.h5", 'r') as f:
                # Load embeddings
                if 'embeddings' in f:
                    result['embeddings'] = f['embeddings'][:]
                
                # Load results
                if 'results' in f:
                    result['results'] = self._load_results_from_hdf5(f['results'])
                
                # Load plots if requested
                if include_plots and 'plots' in f:
                    result['plots'] = self._load_plot_data(f['plots'])
            
            return result
            
        except Exception as e:
            logging.error(f"Error retrieving experiment {experiment_id}: {str(e)}")
            raise
    
    async def get_knowledge_graph(self, 
                                experiment_type: str = None) -> Graph:
        """Retrieve knowledge graph for specific experiment type"""
        try:
            graph_path = self.knowledge_path / f"{experiment_type}_knowledge.ttl"
            if graph_path.exists():
                g = Graph()
                g.parse(str(graph_path), format="turtle")
                return g
            return Graph()
        except Exception as e:
            logging.error(f"Error retrieving knowledge graph: {str(e)}")
            raise
    
    async def generate_plot(self,
                          plot_type: str,
                          data: Dict[str, Any],
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate plot data for frontend visualization"""
        try:
            if plot_type == "scatter":
                fig = px.scatter(
                    data_frame=pd.DataFrame(data),
                    **params
                )
            elif plot_type == "line":
                fig = px.line(
                    data_frame=pd.DataFrame(data),
                    **params
                )
            elif plot_type == "heatmap":
                fig = px.imshow(
                    data,
                    **params
                )
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Convert to JSON for frontend
            plot_json = fig.to_json()
            
            # Cache plot data
            plot_id = f"plot_{int(time.time())}"
            self.redis_client.setex(
                f"plot:{plot_id}",
                3600,  # Cache for 1 hour
                plot_json
            )
            
            return {
                'plot_id': plot_id,
                'plot_data': plot_json
            }
            
        except Exception as e:
            logging.error(f"Error generating plot: {str(e)}")
            raise
    
    def _store_results_in_hdf5(self, group: h5py.Group, results: Dict[str, Any]):
        """Store nested results structure in HDF5"""
        for key, value in results.items():
            if isinstance(value, (np.ndarray, list)):
                group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                self._store_results_in_hdf5(subgroup, value)
            else:
                group.attrs[key] = value
    
    def _load_results_from_hdf5(self, group: h5py.Group) -> Dict[str, Any]:
        """Load nested results structure from HDF5"""
        results = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                results[key] = group[key][:]
            elif isinstance(group[key], h5py.Group):
                results[key] = self._load_results_from_hdf5(group[key])
        
        # Load attributes
        for key, value in group.attrs.items():
            results[key] = value
            
        return results
    
    async def _update_knowledge_graph(self, exp_id: str, data: Dict[str, Any]):
        """Update knowledge graph with new experiment data"""
        exp_type = data['metadata']['type']
        graph_path = self.knowledge_path / f"{exp_type}_knowledge.ttl"
        
        g = Graph()
        if graph_path.exists():
            g.parse(str(graph_path), format="turtle")
        
        # Add new experiment nodes and relationships
        exp_uri = URIRef(f"experiment:{exp_id}")
        g.add((exp_uri, RDF.type, URIRef(f"type:{exp_type}")))
        
        # Add metadata
        for key, value in data['metadata'].items():
            g.add((exp_uri, URIRef(f"metadata:{key}"), Literal(value)))
        
        # Add relationships based on results
        if 'results' in data:
            self._add_result_relationships(g, exp_uri, data['results'])
        
        # Save updated graph
        g.serialize(str(graph_path), format="turtle")
    
    def _cache_experiment_data(self, exp_id: str, data: Dict[str, Any]):
        """Cache frequently accessed experiment data in Redis"""
        cache_data = {
            'metadata': data['metadata'],
            'summary': self._generate_summary(data)
        }
        
        self.redis_client.setex(
            f"exp:{exp_id}",
            3600,  # Cache for 1 hour
            json.dumps(cache_data)
        )
    
    def _get_cached_data(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached experiment data"""
        cached = self.redis_client.get(f"exp:{exp_id}")
        if cached:
            return json.loads(cached)
        return None

    def _store_plot_data(self, group: h5py.Group, plot_name: str, plot_data: Dict[str, Any]):
        """Store plot data in HDF5"""
        group.create_dataset(plot_name, data=plot_data)

    def _load_plot_data(self, group: h5py.Group) -> Dict[str, Any]:
        """Load plot data from HDF5"""
        plot_data = {}
        for key in group.keys():
            plot_data[key] = group[key][:]
        return plot_data

    def _add_result_relationships(self, graph: Graph, exp_uri: URIRef, results: Dict[str, Any]):
        """Add relationships based on experiment results"""
        for key, value in results.items():
            if isinstance(value, (np.ndarray, list)):
                result_uri = URIRef(f"result:{key}")
                graph.add((exp_uri, URIRef(f"hasResult:{key}"), result_uri))
                graph.add((result_uri, RDF.type, URIRef(f"type:{type(value).__name__}")))
                graph.add((result_uri, URIRef(f"value:{key}"), Literal(value)))
            elif isinstance(value, dict):
                subgroup = graph.create_group(f"result:{key}")
                self._add_result_relationships(graph, exp_uri, value)

    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate a summary of experiment data"""
        summary = f"Experiment ID: {data['metadata']['id']}\n"
        summary += f"Type: {data['metadata']['type']}\n"
        summary += f"Summary: {data['metadata'].get('summary', 'No summary provided')}\n"
        return summary 