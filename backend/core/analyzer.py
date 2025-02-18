from typing import Dict, Any
import yaml
import git
import os
from pathlib import Path
import hashlib
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, config_path: str, workspace_dir: str):
        self.config = self._load_config(config_path)
        self.workspace = Path(workspace_dir)
        self.repo = self._initialize_git_repo()
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_git_repo(self) -> git.Repo:
        """Initialize or load Git repository for experiment tracking"""
        if not (self.workspace / '.git').exists():
            repo = git.Repo.init(self.workspace)
            
            # Create .gitignore for large binary files
            gitignore_path = self.workspace / '.gitignore'
            if not gitignore_path.exists():
                with open(gitignore_path, 'w') as f:
                    f.write("*.raw\n*.mzML\n*.fastq\n*.bam\n")
            
            # Initial commit
            repo.index.add(['.gitignore'])
            repo.index.commit("Initial commit")
        else:
            repo = git.Repo(self.workspace)
        
        return repo
    
    def track_changes(self) -> Dict[str, Any]:
        """Monitor changes in the workspace and create a change record"""
        changes = {
            'added': [],
            'modified': [],
            'deleted': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for untracked and modified files
        for item in self.repo.index.diff(None):
            if item.change_type == 'A':
                changes['added'].append(item.a_path)
            elif item.change_type == 'M':
                changes['modified'].append(item.a_path)
            elif item.change_type == 'D':
                changes['deleted'].append(item.a_path)
        
        # Generate checksums for important files
        for file_path in self.workspace.rglob('*'):
            if file_path.is_file() and not any(pattern in str(file_path) for pattern in ['.git', '.gitignore']):
                changes['checksums'] = self._calculate_checksum(file_path)
        
        return changes
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def commit_experiment_state(self, message: str = None):
        """Commit current state of the experiment"""
        if not message:
            message = f"Experiment state update: {datetime.now().isoformat()}"
        
        # Stage all tracked files
        self.repo.git.add(update=True)
        
        # Add new files that aren't in .gitignore
        self.repo.git.add('.')
        
        # Commit if there are changes
        if self.repo.index.diff('HEAD'):
            self.repo.index.commit(message)
    
    def run(self):
        """Run the experiment analysis pipeline"""
        experiment_type = self.config['type']
        
        # Record initial state
        initial_state = self.track_changes()
        
        try:
            # Run appropriate pipeline
            if experiment_type == 'genomics':
                result = self._run_genomics()
            elif experiment_type == 'metabolomics':
                result = self._run_metabolomics()
            
            # Record final state and changes
            final_state = self.track_changes()
            
            # Commit the experiment results
            self.commit_experiment_state(
                f"Completed {experiment_type} experiment: {self.config.get('name', 'unnamed')}"
            )
            
            return {
                'status': 'success',
                'initial_state': initial_state,
                'final_state': final_state,
                'result': result
            }
            
        except Exception as e:
            # Record error state
            error_state = self.track_changes()
            self.commit_experiment_state(f"Error in {experiment_type} experiment: {str(e)}")
            raise
    
    def _run_genomics(self):
        # Implement genomics-specific pipeline
        pass
    
    def _run_metabolomics(self):
        # Implement metabolomics-specific pipeline
        pass
