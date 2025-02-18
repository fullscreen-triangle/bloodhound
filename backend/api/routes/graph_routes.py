from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from rdflib import Graph, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery

router = APIRouter()

@router.get("/knowledge/{experiment_type}")
async def get_knowledge_graph(experiment_type: str):
    """Retrieve knowledge graph for specific experiment type"""
    try:
        graph = await db_manager.get_knowledge_graph(experiment_type)
        return {
            'nodes': _extract_nodes(graph),
            'edges': _extract_edges(graph)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/query")
async def query_knowledge_graph(query_data: Dict[str, Any]):
    """Execute SPARQL query on knowledge graph"""
    try:
        results = await db_manager.query_knowledge_graph(
            experiment_type=query_data['experiment_type'],
            query=query_data['sparql_query']
        )
        return {'results': results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _extract_nodes(graph: Graph) -> List[Dict[str, Any]]:
    """Extract nodes from RDF graph in visualization format"""
    nodes = []
    for s in graph.subjects():
        if isinstance(s, URIRef):
            node_type = str(s).split(':')[0]
            nodes.append({
                'id': str(s),
                'type': node_type,
                'label': str(s).split(':')[-1]
            })
    return nodes

def _extract_edges(graph: Graph) -> List[Dict[str, Any]]:
    """Extract edges from RDF graph in visualization format"""
    edges = []
    for s, p, o in graph:
        if isinstance(o, URIRef):
            edges.append({
                'source': str(s),
                'target': str(o),
                'label': str(p).split('#')[-1]
            })
    return edges 