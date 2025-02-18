from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from ...core.db_manager import DistributedDBManager
from ...utils.visualization.graph_viz import GraphVisualizer

router = APIRouter()
db_manager = DistributedDBManager()
graph_viz = GraphVisualizer()

@router.post("/plot")
async def generate_plot(plot_data: Dict[str, Any]):
    """Generate plot from experiment data"""
    try:
        plot = await db_manager.generate_plot(
            plot_type=plot_data['type'],
            data=plot_data['data'],
            params=plot_data.get('params', {})
        )
        return plot
    except Exception as e:
        logging.error(f"Plot generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plot/{plot_id}")
async def get_plot(plot_id: str):
    """Retrieve cached plot"""
    try:
        plot_data = await db_manager.get_cached_plot(plot_id)
        if not plot_data:
            raise HTTPException(status_code=404, detail="Plot not found")
        return plot_data
    except Exception as e:
        logging.error(f"Plot retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/{experiment_type}")
async def get_knowledge_graph(experiment_type: str):
    """Get knowledge graph visualization"""
    try:
        graph = await db_manager.get_knowledge_graph(experiment_type)
        return graph_viz.create_network_graph(
            nodes=graph['nodes'],
            edges=graph['edges']
        )
    except Exception as e:
        logging.error(f"Knowledge graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/query")
async def query_knowledge_graph(query_data: Dict[str, Any]):
    """Query knowledge graph"""
    try:
        results = await db_manager.query_knowledge_graph(
            experiment_type=query_data['experiment_type'],
            query=query_data['sparql_query']
        )
        return {'results': results}
    except Exception as e:
        logging.error(f"Knowledge graph query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 