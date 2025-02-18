import psutil
import torch
import platform
from typing import Dict, Any
import subprocess
import multiprocessing
from enum import Enum

class ComputeEnvironment(Enum):
    LOCAL = "local"
    HPC = "hpc"
    DISTRIBUTED = "distributed"

class ResourceManager:
    def __init__(self):
        self.system_specs = self._analyze_system()
        self.compute_env = self._determine_environment()
        
    def _analyze_system(self) -> Dict[str, Any]:
        """Analyze available system resources"""
        specs = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_info': [],
            'disk_space_gb': psutil.disk_usage('/').total / (1024**3),
            'platform': platform.system(),
            'architecture': platform.machine()
        }
        
        # Get GPU details if available
        if specs['gpu_available']:
            for i in range(specs['gpu_count']):
                specs['gpu_info'].append({
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory
                })
                
        # Check for SLURM (HPC environment)
        try:
            subprocess.run(['sinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            specs['slurm_available'] = True
        except FileNotFoundError:
            specs['slurm_available'] = False
            
        return specs
    
    def _determine_environment(self) -> ComputeEnvironment:
        """Determine the best compute environment based on specs"""
        if self.system_specs['slurm_available']:
            return ComputeEnvironment.HPC
        elif self.system_specs['memory_gb'] < 16:  # Low resource system
            return ComputeEnvironment.DISTRIBUTED
        else:
            return ComputeEnvironment.LOCAL
            
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration based on available resources"""
        config = {
            'batch_size': self._calculate_batch_size(),
            'num_workers': self._calculate_workers(),
            'use_gpu': self.system_specs['gpu_available'],
            'distributed': self.compute_env == ComputeEnvironment.DISTRIBUTED,
            'model_size': self._determine_model_size()
        }
        return config
    
    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        if self.system_specs['gpu_available']:
            # GPU-based calculation
            return min(32, int(self.system_specs['gpu_info'][0]['memory'] / (1024**3) * 4))
        else:
            # CPU-based calculation
            return min(16, int(self.system_specs['memory_gb'] / 4))
    
    def _calculate_workers(self) -> int:
        """Calculate optimal number of worker processes"""
        return min(4, self.system_specs['cpu_count'] - 1)
    
    def _determine_model_size(self) -> str:
        """Determine appropriate model size based on resources"""
        if self.system_specs['memory_gb'] < 8:
            return 'tiny'
        elif self.system_specs['memory_gb'] < 16:
            return 'small'
        else:
            return 'base' 