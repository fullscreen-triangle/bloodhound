from typing import Dict, Any, List, Optional, Set
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
import networkx as nx
import git
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class FileNode:
    """Represents a file in the tracking network"""
    path: str
    hash: str
    machine_id: str
    timestamp: float
    metadata: Dict[str, Any]
    is_raw_data: bool
    parent_hashes: Set[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'hash': self.hash,
            'machine_id': self.machine_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'is_raw_data': self.is_raw_data,
            'parent_hashes': list(self.parent_hashes) if self.parent_hashes else []
        }

class ChangeTracker:
    """Tracks changes to files in a distributed system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.machine_id = self.config.get('machine_id', self._generate_machine_id())
        self.network = nx.DiGraph()
        self.load_network()
        
    def track_file(self, 
                  file_path: Path,
                  is_raw_data: bool = False,
                  parent_hashes: Optional[Set[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> FileNode:
        """Track a file and its changes"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            
            node = FileNode(
                path=str(file_path),
                hash=file_hash,
                machine_id=self.machine_id,
                timestamp=datetime.utcnow().timestamp(),
                metadata=metadata or {},
                is_raw_data=is_raw_data,
                parent_hashes=parent_hashes or set()
            )
            
            # Add to network
            self.network.add_node(file_hash, **node.to_dict())
            
            # Add edges from parents
            if parent_hashes:
                for parent_hash in parent_hashes:
                    if parent_hash in self.network:
                        self.network.add_edge(parent_hash, file_hash)
            
            self.save_network()
            return node
            
        except Exception as e:
            logging.error(f"File tracking error: {str(e)}")
            raise
            
    def detect_changes(self, directory: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Detect changes in tracked files"""
        try:
            changes = {
                'modified': [],
                'new': [],
                'missing': []
            }
            
            # Get all tracked files in directory
            tracked_files = {
                node['path']: node 
                for _, node in self.network.nodes(data=True)
                if Path(node['path']).is_relative_to(directory)
            }
            
            # Check current files
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    current_hash = self._calculate_file_hash(file_path)
                    str_path = str(file_path)
                    
                    if str_path in tracked_files:
                        if current_hash != tracked_files[str_path]['hash']:
                            changes['modified'].append({
                                'path': str_path,
                                'old_hash': tracked_files[str_path]['hash'],
                                'new_hash': current_hash
                            })
                    else:
                        changes['new'].append({
                            'path': str_path,
                            'hash': current_hash
                        })
            
            # Check for missing files
            current_files = set(str(p) for p in directory.rglob('*') if p.is_file())
            missing_files = set(tracked_files.keys()) - current_files
            changes['missing'].extend([{
                'path': path,
                'hash': tracked_files[path]['hash']
            } for path in missing_files])
            
            return changes
            
        except Exception as e:
            logging.error(f"Change detection error: {str(e)}")
            raise
            
    def get_file_history(self, file_hash: str) -> List[Dict[str, Any]]:
        """Get the history of a file"""
        try:
            if file_hash not in self.network:
                return []
                
            # Get all predecessors (parents)
            predecessors = nx.ancestors(self.network, file_hash)
            predecessors.add(file_hash)
            
            # Create subgraph of file history
            history_graph = self.network.subgraph(predecessors)
            
            # Convert to list of nodes with attributes
            history = [
                {**node_attr, 'hash': node_hash}
                for node_hash, node_attr in history_graph.nodes(data=True)
            ]
            
            # Sort by timestamp
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logging.error(f"File history error: {str(e)}")
            raise
    
    def merge_networks(self, other_network: nx.DiGraph):
        """Merge another network into this one"""
        try:
            # Add all nodes and edges from other network
            self.network.add_nodes_from(other_network.nodes(data=True))
            self.network.add_edges_from(other_network.edges())
            self.save_network()
            
        except Exception as e:
            logging.error(f"Network merge error: {str(e)}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _generate_machine_id(self) -> str:
        """Generate unique machine identifier"""
        import socket
        import uuid
        
        # Combine hostname and hardware identifier
        machine_id = f"{socket.gethostname()}-{uuid.getnode()}"
        return hashlib.md5(machine_id.encode()).hexdigest()
    
    def save_network(self):
        """Save network to file"""
        network_path = Path(self.config.get('network_path', 'data/tracking/network.json'))
        network_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert network to serializable format
        data = {
            'nodes': [
                {**attr, 'id': node_id}
                for node_id, attr in self.network.nodes(data=True)
            ],
            'edges': list(self.network.edges())
        }
        
        with open(network_path, 'w') as f:
            json.dump(data, f)
    
    def load_network(self):
        """Load network from file"""
        network_path = Path(self.config.get('network_path', 'data/tracking/network.json'))
        
        if network_path.exists():
            with open(network_path, 'r') as f:
                data = json.load(f)
                
            # Reconstruct network
            self.network = nx.DiGraph()
            for node in data['nodes']:
                node_id = node.pop('id')
                self.network.add_node(node_id, **node)
            self.network.add_edges_from(data['edges']) 