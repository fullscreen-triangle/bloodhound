from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from cryptography.fernet import Fernet
import logging

class FederatedModel(ABC):
    """Abstract base class for federated learning models"""
    
    @abstractmethod
    def train_local(self, data: Dict[str, Any], epochs: int) -> None:
        """Train model on local data"""
        pass
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters"""
        pass

class PrivacyEngine:
    """Privacy preservation for federated learning"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        
    def add_noise(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to parameters"""
        try:
            noisy_parameters = {}
            
            for name, param in parameters.items():
                sensitivity = torch.norm(param) / len(param.view(-1))
                noise_scale = sensitivity / self.epsilon
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=param.size(),
                    device=param.device
                )
                noisy_parameters[name] = param + noise
                
            return noisy_parameters
            
        except Exception as e:
            logging.error(f"Privacy error: {str(e)}")
            raise

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def encrypt_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt model parameters"""
        try:
            param_bytes = self._serialize_parameters(parameters)
            return self.cipher_suite.encrypt(param_bytes)
        except Exception as e:
            logging.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt_parameters(self, encrypted_params: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters"""
        try:
            param_bytes = self.cipher_suite.decrypt(encrypted_params)
            return self._deserialize_parameters(param_bytes)
        except Exception as e:
            logging.error(f"Decryption error: {str(e)}")
            raise
    
    def _serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        """Serialize parameters to bytes"""
        return torch.save(parameters, buffer=bytes())
    
    def _deserialize_parameters(self, param_bytes: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize parameters from bytes"""
        return torch.load(param_bytes) 