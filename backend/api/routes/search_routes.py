from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from ...core.embedder import TextEmbedder
from ...core.ai_chat import AIChatModel
from ...core.db_manager import DistributedDBManager

router = APIRouter()
embedder = TextEmbedder()
chat_model = AIChatModel()
db_manager = DistributedDBManager()

@router.post("/query")
async def search(query_data: Dict[str, Any]):
    """Search across experiments, knowledge graphs, and embeddings"""
    try:
        query_embedding = embedder.encode_text([query_data['query']])[0]
        results = []
        
        # Search in specified sources
        sources = query_data.get('sources', ['experiments', 'knowledge_graph', 'embeddings'])
        
        if 'experiments' in sources:
            results.extend(await db_manager.search_experiments(
                query=query_data['query'],
                embedding=query_embedding,
                filters=query_data.get('filters', {})
            ))
        
        if 'knowledge_graph' in sources:
            results.extend(await db_manager.search_knowledge_graph(
                query=query_data['query']
            ))
        
        if 'embeddings' in sources:
            results.extend(await db_manager.search_similar(
                embedding=query_embedding,
                limit=query_data.get('limit', 10)
            ))
        
        # Enhance results with LLM
        enhanced_results = await chat_model.enhance_search_results(
            query=query_data['query'],
            results=results
        )
        
        return enhanced_results
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggestions")
async def get_suggestions(query: str):
    """Get search suggestions"""
    try:
        suggestions = []
        suggestions.extend(await db_manager.get_suggestions(query))
        suggestions.extend(await db_manager.get_graph_suggestions(query))
        return suggestions
    except Exception as e:
        logging.error(f"Suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 