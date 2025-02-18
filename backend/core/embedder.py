from typing import List, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .federated.base import FederatedModel
import logging

class TextEmbedder(FederatedModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Error encoding text: {str(e)}")
            raise
            
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling of token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def train_local(self, data: Dict[str, Any], epochs: int) -> None:
        """Train model on local data"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters())
            
            for epoch in range(epochs):
                for batch in data['train_loader']:
                    texts, labels = batch
                    embeddings = self.encode_text(texts)
                    loss = self._compute_contrastive_loss(embeddings, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        except Exception as e:
            logging.error(f"Error in local training: {str(e)}")
            raise
            
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for aggregation"""
        return {name: param.data for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters after aggregation"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
                    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data['val_loader']:
                texts, labels = batch
                embeddings = self.encode_text(texts)
                loss = self._compute_contrastive_loss(embeddings, labels)
                total_loss += loss.item()
                num_batches += 1
                
        return {'avg_loss': total_loss / num_batches}
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for embeddings"""
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        labels_matrix = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        positive_pairs = similarity_matrix * labels_matrix
        negative_pairs = similarity_matrix * (1 - labels_matrix)
        
        loss = -torch.log(
            torch.exp(positive_pairs) / 
            (torch.exp(positive_pairs) + torch.sum(torch.exp(negative_pairs), dim=1))
        ).mean()
        
        return loss
