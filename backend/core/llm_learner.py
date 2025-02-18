from typing import Dict, Any, List
import torch
import logging
from datetime import datetime
from .federated.base import FederatedModel
from .embedder import TextEmbedder

class ContinuousLearner:
    def __init__(self, base_model: FederatedModel):
        self.model = base_model
        self.embedder = TextEmbedder()
        self.interaction_buffer = []
        self.learning_threshold = 50  # Number of interactions before learning
        self.max_buffer_size = 1000
        
    async def learn_from_interaction(self, 
                                   query: str, 
                                   response: str, 
                                   context: Dict[str, Any],
                                   feedback: Dict[str, Any] = None):
        """Learn from each API interaction"""
        try:
            # Create learning instance
            learning_instance = {
                'query': query,
                'response': response,
                'context': context,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat(),
                'embedding': self.embedder.encode_text([query])[0]
            }
            
            # Add to buffer
            self.interaction_buffer.append(learning_instance)
            
            # Trigger learning if threshold reached
            if len(self.interaction_buffer) >= self.learning_threshold:
                await self._perform_learning_step()
                
            # Maintain buffer size
            if len(self.interaction_buffer) > self.max_buffer_size:
                self._prune_buffer()
                
        except Exception as e:
            logging.error(f"Learning error: {str(e)}")
            raise
            
    async def _perform_learning_step(self):
        """Perform incremental learning on collected data"""
        try:
            # Prepare training data
            training_data = self._prepare_training_data()
            
            # Update model
            await self.model.train_local(
                data=training_data,
                epochs=1  # Incremental learning
            )
            
            # Clear buffer after learning
            self.interaction_buffer = []
            
            logging.info("Completed incremental learning step")
            
        except Exception as e:
            logging.error(f"Learning step failed: {str(e)}")
            raise
            
    def _prepare_training_data(self) -> Dict[str, torch.Tensor]:
        """Prepare interaction data for training"""
        queries = []
        responses = []
        contexts = []
        
        for instance in self.interaction_buffer:
            queries.append(instance['query'])
            responses.append(instance['response'])
            contexts.append(instance['context'])
            
        return {
            'queries': self.embedder.encode_text(queries),
            'responses': self.embedder.encode_text(responses),
            'contexts': contexts
        }
        
    def _prune_buffer(self):
        """Remove oldest or least valuable interactions"""
        # Sort by timestamp and feedback score
        self.interaction_buffer.sort(
            key=lambda x: (
                x.get('feedback', {}).get('value', 0),
                x['timestamp']
            )
        )
        
        # Keep only the most recent/valuable instances
        self.interaction_buffer = self.interaction_buffer[-self.max_buffer_size:] 