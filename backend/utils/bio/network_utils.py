import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd
from scipy import stats

class BiologicalNetwork:
    """Build and analyze biological networks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph = nx.Graph()
        
    def build_network(self, 
                     nodes: List[Dict[str, Any]], 
                     edges: List[Dict[str, Any]]) -> nx.Graph:
        """Build network from nodes and edges"""
        try:
            # Add nodes with attributes
            for node in nodes:
                self.graph.add_node(
                    node['id'],
                    **{k: v for k, v in node.items() if k != 'id'}
                )
            
            # Add edges with attributes
            for edge in edges:
                self.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    **{k: v for k, v in edge.items() 
                       if k not in ['source', 'target']}
                )
                
            return self.graph
            
        except Exception as e:
            logging.error(f"Network building error: {str(e)}")
            raise
            
    def analyze_network(self) -> Dict[str, Any]:
        """Analyze network properties"""
        try:
            analysis = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'degree_centrality': dict(nx.degree_centrality(self.graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(self.graph)),
                'connected_components': list(nx.connected_components(self.graph))
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Network analysis error: {str(e)}")
            raise
            
    def find_modules(self, 
                    algorithm: str = 'louvain') -> List[Dict[str, Any]]:
        """Find network modules/communities"""
        try:
            if algorithm == 'louvain':
                from community import community_louvain
                partition = community_louvain.best_partition(self.graph)
            elif algorithm == 'leiden':
                from leidenalg import find_partition
                partition = find_partition(self.graph)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
            # Group nodes by module
            modules = {}
            for node, module_id in partition.items():
                if module_id not in modules:
                    modules[module_id] = []
                modules[module_id].append(node)
                
            return [
                {
                    'id': module_id,
                    'nodes': nodes,
                    'size': len(nodes)
                }
                for module_id, nodes in modules.items()
            ]
            
        except Exception as e:
            logging.error(f"Module detection error: {str(e)}")
            raise
            
    def calculate_node_importance(self, 
                                method: str = 'pagerank') -> Dict[str, float]:
        """Calculate node importance scores"""
        try:
            if method == 'pagerank':
                return nx.pagerank(self.graph)
            elif method == 'eigenvector':
                return nx.eigenvector_centrality(self.graph)
            elif method == 'katz':
                return nx.katz_centrality(self.graph)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logging.error(f"Node importance calculation error: {str(e)}")
            raise
