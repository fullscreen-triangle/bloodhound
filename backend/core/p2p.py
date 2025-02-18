from typing import Dict, Any, List, Optional, Set
import asyncio
import logging
from dataclasses import dataclass
import time
import random
from cryptography.fernet import Fernet
import socket
import json
import torch

@dataclass
class Peer:
    id: str
    address: str
    port: int
    last_seen: float
    capabilities: Dict[str, Any]

class P2PNode:
    """Peer-to-peer node for distributed computation"""
    
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.node_id = self._generate_node_id()
        self.peers: Dict[str, Peer] = {}
        self.active_rounds: Dict[str, Set[str]] = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize network
        self.host = self._get_local_ip()
        self.port = self._find_available_port()
        self.server = None
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        return f"node_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    
    def _find_available_port(self) -> int:
        """Find available port for P2P communication"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    async def start(self):
        """Start P2P node"""
        self.server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        logging.info(f"P2P node started at {self.host}:{self.port}")
    
    async def _handle_connection(self, 
                               reader: asyncio.StreamReader, 
                               writer: asyncio.StreamWriter):
        """Handle incoming P2P connections"""
        try:
            data = await reader.read()
            message = self.cipher_suite.decrypt(data)
            response = await self._process_message(json.loads(message))
            
            writer.write(self.cipher_suite.encrypt(json.dumps(response).encode()))
            await writer.drain()
            
        except Exception as e:
            logging.error(f"Error handling connection: {str(e)}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def participate_in_aggregation(self,
                                       round_id: str,
                                       local_params: bytes) -> bytes:
        """Participate in parameter aggregation round"""
        try:
            # Register in round
            self.active_rounds[round_id] = set()
            
            # Share parameters with peers
            tasks = []
            for peer_id, peer in self.peers.items():
                if self._is_peer_eligible(peer):
                    task = self._share_parameters(peer, round_id, local_params)
                    tasks.append(task)
            
            # Wait for responses
            responses = await asyncio.gather(*tasks)
            
            # Aggregate parameters
            aggregated = self._aggregate_parameters(responses)
            return aggregated
            
        except Exception as e:
            logging.error(f"Aggregation failed: {str(e)}")
            raise
    
    def _is_peer_eligible(self, peer: Peer) -> bool:
        """Check if peer is eligible for participation"""
        return (time.time() - peer.last_seen < 300 and  # 5 minutes timeout
                peer.capabilities.get('can_aggregate', False))
    
    async def _share_parameters(self,
                              peer: Peer,
                              round_id: str,
                              parameters: bytes) -> bytes:
        """Share parameters with a peer"""
        try:
            reader, writer = await asyncio.open_connection(
                peer.address, peer.port
            )
            
            # Send parameters
            message = {
                'type': 'parameters',
                'round_id': round_id,
                'data': parameters
            }
            writer.write(self.cipher_suite.encrypt(json.dumps(message).encode()))
            await writer.drain()
            
            # Get response
            response = await reader.read()
            return self.cipher_suite.decrypt(response)
            
        except Exception as e:
            logging.error(f"Error sharing parameters with {peer.id}: {str(e)}")
            return None
        finally:
            writer.close()
            await writer.wait_closed()
    
    def _aggregate_parameters(self, parameter_list: List[bytes]) -> bytes:
        """Aggregate parameters from multiple peers"""
        # Filter out None responses
        valid_params = [p for p in parameter_list if p is not None]
        
        if not valid_params:
            raise ValueError("No valid parameters received for aggregation")
        
        # Deserialize and aggregate
        aggregated = {}
        num_params = len(valid_params)
        
        for params in valid_params:
            param_dict = torch.load(params)
            for name, param in param_dict.items():
                if name not in aggregated:
                    aggregated[name] = param
                else:
                    aggregated[name] += param
        
        # Average parameters
        for name in aggregated:
            aggregated[name] /= num_params
        
        # Serialize and return
        return torch.save(aggregated, buffer=bytes())
