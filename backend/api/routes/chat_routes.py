from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from ...core.ai_chat import AIChatModel
from ...core.llm_learner import ContinuousLearner
from ...core.db_manager import DistributedDBManager

router = APIRouter()
chat_model = AIChatModel()
learner = ContinuousLearner(chat_model)
db_manager = DistributedDBManager()

@router.post("/message")
async def handle_chat_message(message_data: Dict[str, Any]):
    """Handle chat messages with continuous learning"""
    try:
        query = message_data['message']
        context = message_data.get('context', {})
        experiment_id = message_data.get('experimentId')
        
        # Get experiment context if ID provided
        if experiment_id:
            exp_context = await db_manager.retrieve_experiment(experiment_id)
            context.update(exp_context)
        
        # Get relevant knowledge from database
        knowledge_context = await db_manager.get_relevant_knowledge(
            query=query,
            experiment_id=experiment_id
        )
        
        # Generate response
        response = await chat_model.generate_response(
            query=query,
            context={**context, **knowledge_context}
        )
        
        # Learn from interaction
        await learner.learn_from_interaction(
            query=query,
            response=response['message'],
            context=knowledge_context
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def handle_chat_feedback(feedback_data: Dict[str, Any]):
    """Handle user feedback for continuous learning"""
    try:
        # Store feedback
        await db_manager.store_feedback(feedback_data)
        
        # Update learning instance with feedback
        await learner.learn_from_interaction(
            query=feedback_data['query'],
            response=feedback_data['response'],
            context=feedback_data['context'],
            feedback={
                'value': feedback_data['rating'],
                'comments': feedback_data.get('comments')
            }
        )
        
        return {"status": "success"}
        
    except Exception as e:
        logging.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 