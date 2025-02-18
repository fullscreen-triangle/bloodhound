import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import logging
import json

class GraphVisualizer:
    """Visualization utilities for network graphs"""
    
    @staticmethod
    def create_network_graph(nodes: List[Dict[str, Any]], 
                           edges: List[Dict[str, Any]],
                           layout: str = "force") -> Dict[str, Any]:
        """Create interactive network visualization"""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes with attributes
            for node in nodes:
                G.add_node(
                    node['id'],
                    **{k: v for k, v in node.items() if k != 'id'}
                )
            
            # Add edges with attributes
            for edge in edges:
                G.add_edge(
                    edge['source'],
                    edge['target'],
                    **{k: v for k, v in edge.items() 
                       if k not in ['source', 'target']}
                )
            
            # Calculate layout
            if layout == "force":
                pos = nx.spring_layout(G)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Create Plotly trace for nodes
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                hoverinfo='text',
                text=[G.nodes[node].get('label', node) for node in G.nodes()],
                marker=dict(
                    size=15,
                    color=[G.nodes[node].get('color', '#1f77b4') 
                          for node in G.nodes()]
                )
            )
            
            # Create Plotly trace for edges
            edge_trace = go.Scatter(
                x=[],
                y=[],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            )
            
            # Add edges to trace
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            return {
                'plotly_json': json.dumps(fig.to_dict()),
                'network_data': {
                    'nodes': list(G.nodes(data=True)),
                    'edges': list(G.edges(data=True))
                }
            }
            
        except Exception as e:
            logging.error(f"Graph visualization error: {str(e)}")
            raise
    
    @staticmethod
    def create_knowledge_graph(
            triples: List[tuple],
            highlight_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create visualization for knowledge graph"""
        try:
            G = nx.DiGraph()
            
            # Add edges from triples (subject, predicate, object)
            for s, p, o in triples:
                G.add_edge(s, o, relation=p)
            
            # Calculate layout
            pos = nx.spring_layout(G)
            
            # Create node trace
            node_colors = []
            for node in G.nodes():
                if highlight_nodes and node in highlight_nodes:
                    node_colors.append('#ff7f0e')  # Highlighted color
                else:
                    node_colors.append('#1f77b4')  # Default color
            
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                hoverinfo='text',
                text=list(G.nodes()),
                marker=dict(
                    size=15,
                    color=node_colors
                )
            )
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=[],
                y=[],
                mode='lines+text',
                line=dict(width=1, color='#888'),
                text=[],  # Will contain edge labels
                textposition='middle'
            )
            
            # Add edges and their labels
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
                edge_trace['text'].append(edge[2]['relation'])
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Knowledge Graph Visualization',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40)
                )
            )
            
            return {
                'plotly_json': json.dumps(fig.to_dict()),
                'network_data': {
                    'nodes': list(G.nodes(data=True)),
                    'edges': list(G.edges(data=True))
                }
            }
            
        except Exception as e:
            logging.error(f"Knowledge graph visualization error: {str(e)}")
            raise
