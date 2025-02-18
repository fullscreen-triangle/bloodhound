from typing import Dict, Any, Optional
import torch
import logging
from .base import FederatedModel, PrivacyEngine, SecureAggregation
from ..p2p import P2PNode

class FederatedClient:
    """Client node in federated learning system"""
    
    def __init__(self, 
                 model_type: str,
                 resource_manager: Any,
                 privacy_budget: float = 1.0):
        self.model = self._initialize_model(model_type)
        self.privacy_engine = PrivacyEngine(epsilon=privacy_budget)
        self.secure_aggregation = SecureAggregation()
        self.p2p_node = P2PNode(resource_manager)
        
    def _initialize_model(self, model_type: str) -> FederatedModel:
        """Initialize appropriate model based on type"""
        if model_type == "genomics":
            from .models.genomics import GenomicsModel
            return GenomicsModel()
        elif model_type == "metabolomics":
            from .models.metabolomics import MetabolomicsModel
            return MetabolomicsModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    async def train_local_model(self, 
                               data: Dict[str, Any],
                               config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Train model on local data"""
        try:
            # Train model
            self.model.train_local(
                data=data,
                epochs=config['local_epochs']
            )
            
            # Get parameters and add noise for privacy
            parameters = self.model.get_parameters()
            private_parameters = self.privacy_engine.add_noise(parameters)
            
            return private_parameters
            
        except Exception as e:
            logging.error(f"Local training failed: {str(e)}")
            raise
    
    async def participate_in_aggregation(self,
                                       local_model: Dict[str, torch.Tensor],
                                       round_id: str) -> Dict[str, torch.Tensor]:
        """Participate in secure aggregation round"""
        try:
            # Encrypt local parameters
            encrypted_params = self.secure_aggregation.encrypt_parameters(local_model)
            
            # Share through P2P network
            aggregated_params = await self.p2p_node.participate_in_aggregation(
                round_id=round_id,
                local_params=encrypted_params
            )
            
            # Decrypt and update local model
            decrypted_params = self.secure_aggregation.decrypt_parameters(aggregated_params)
            self.model.set_parameters(decrypted_params)
            
            return decrypted_params
            
        except Exception as e:
            logging.error(f"Aggregation participation failed: {str(e)}")
            raise 