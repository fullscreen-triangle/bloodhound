from typing import Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
import subprocess
import sys
import signal
from utils.task.task_manager import TaskManager
from utils.tracking.file_tracker import ChangeTracker
from utils.config.config_manager import ConfigManager
from pipelines.genomics_manager import GenomicsPipelineManager
from pipelines.metabolomics_manager import MetabolomicsPipelineManager
from core.federated.client import FederatedClient
from core.federated.base import PrivacyEngine, SecureAggregation
from fastapi import FastAPI
import uvicorn
from multiprocessing import Process
import webbrowser
import json
import os
import time

class ApplicationManager:
    """Manages the entire application including frontend, backend, and federated learning"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = Path(__file__).parent.parent
        self.load_config(config_path)
        self.setup_logging()
        
        # Initialize managers
        self.config_manager = ConfigManager(self.root_dir)
        self.task_manager = TaskManager(self.config.get('task_manager', {}))
        self.change_tracker = ChangeTracker(self.config.get('change_tracker', {}))
        
        # Initialize resource manager
        self.resource_manager = self._initialize_resource_manager()
        
        # Initialize federated learning components
        self.federated_clients = {}
        self._initialize_federated_learning()
        
        # Initialize pipeline managers with federated clients
        self.pipeline_managers = self._initialize_pipeline_managers()
        
        self.backend_process = None
        self.frontend_process = None
        self.is_running = False
        
    def _initialize_resource_manager(self):
        """Initialize resource manager for hardware optimization"""
        # Implementation of resource management
        pass
        
    def _initialize_federated_learning(self):
        """Initialize federated learning clients for each experiment type"""
        try:
            fl_config = self.config.get('federated_learning', {})
            
            for exp_type in ['genomics', 'metabolomics']:
                self.federated_clients[exp_type] = FederatedClient(
                    model_type=exp_type,
                    resource_manager=self.resource_manager,
                    privacy_budget=fl_config.get('privacy_budget', 1.0)
                )
                
            logging.info("Federated learning components initialized")
            
        except Exception as e:
            logging.error(f"Federated learning initialization error: {str(e)}")
            raise
            
    def _initialize_pipeline_managers(self) -> Dict[str, Any]:
        """Initialize pipeline managers with federated clients"""
        return {
            'genomics': GenomicsPipelineManager(
                resource_manager=self.resource_manager,
                federated_client=self.federated_clients['genomics']
            ),
            'metabolomics': MetabolomicsPipelineManager(
                resource_manager=self.resource_manager,
                federated_client=self.federated_clients['metabolomics']
            )
        }
        
    def get_pipeline_manager(self, experiment_type: str):
        """Get appropriate pipeline manager for experiment type"""
        if experiment_type not in self.pipeline_managers:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        return self.pipeline_managers[experiment_type]
        
    async def participate_in_federated_learning(self, 
                                              experiment_type: str,
                                              local_data: Dict[str, Any]) -> Dict[str, Any]:
        """Participate in federated learning round"""
        try:
            client = self.federated_clients.get(experiment_type)
            if not client:
                raise ValueError(f"No federated client for {experiment_type}")
                
            # Train local model
            local_model = await client.train_local_model(
                data=local_data,
                config=self.config['federated_learning']
            )
            
            # Participate in secure aggregation
            round_id = f"{experiment_type}_{int(time.time())}"
            global_model = await client.participate_in_aggregation(
                local_model=local_model,
                round_id=round_id
            )
            
            return {
                "round_id": round_id,
                "status": "completed",
                "model_updated": True
            }
            
        except Exception as e:
            logging.error(f"Federated learning participation error: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
            
    async def create_experiment(self, 
                              experiment_type: str, 
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and run a new experiment with federated learning"""
        try:
            # Validate against template
            template = self.config_manager.get_template(experiment_type)
            if not template:
                raise ValueError(f"No template found for {experiment_type}")
                
            # Get appropriate pipeline manager
            pipeline_manager = self.get_pipeline_manager(experiment_type)
            
            # Submit experiment task
            task_id = await self.task_manager.submit_task(
                name=f"run_{experiment_type}_experiment",
                func=pipeline_manager.run_analysis,
                args=(input_data, f"{experiment_type}_{task_id}"),
                priority=TaskPriority.HIGH
            )
            
            # Schedule federated learning participation
            await self.task_manager.submit_task(
                name=f"federated_learning_{experiment_type}",
                func=self.participate_in_federated_learning,
                args=(experiment_type, input_data),
                priority=TaskPriority.MEDIUM,
                dependencies=[task_id]
            )
            
            return {
                "task_id": task_id,
                "experiment_type": experiment_type,
                "status": "submitted",
                "federated_learning": "scheduled"
            }
            
        except Exception as e:
            logging.error(f"Experiment creation error: {str(e)}")
            raise
            
    def load_config(self, config_path: Optional[str] = None):
        """Load application configuration"""
        default_config = {
            'backend': {
                'host': 'localhost',
                'port': 8000,
                'reload': True
            },
            'frontend': {
                'host': 'localhost',
                'port': 3000,
                'dev_mode': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log'
            },
            'data_dir': 'data',
            'auto_open_browser': True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config = {**default_config, **user_config}
        else:
            self.config = default_config
            
    def setup_logging(self):
        """Configure application logging"""
        log_config = self.config['logging']
        log_path = Path(log_config['file'])
        log_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        try:
            backend_config = self.config['backend']
            
            def run_backend():
                uvicorn.run(
                    "main:app",
                    host=backend_config['host'],
                    port=backend_config['port'],
                    reload=backend_config['reload']
                )
                
            self.backend_process = Process(target=run_backend)
            self.backend_process.start()
            logging.info(f"Backend started on http://{backend_config['host']}:{backend_config['port']}")
            
        except Exception as e:
            logging.error(f"Backend startup error: {str(e)}")
            raise
            
    def start_frontend(self):
        """Start the React frontend development server"""
        try:
            frontend_config = self.config['frontend']
            frontend_path = Path('frontend')
            
            if frontend_config['dev_mode']:
                # Start development server
                self.frontend_process = subprocess.Popen(
                    ['npm', 'start'],
                    cwd=frontend_path,
                    env={
                        **os.environ,
                        'PORT': str(frontend_config['port']),
                        'REACT_APP_API_URL': f"http://{self.config['backend']['host']}:{self.config['backend']['port']}"
                    }
                )
            else:
                # Serve built frontend
                self.frontend_process = subprocess.Popen(
                    ['serve', '-s', 'build', '-l', str(frontend_config['port'])],
                    cwd=frontend_path
                )
                
            logging.info(f"Frontend started on http://{frontend_config['host']}:{frontend_config['port']}")
            
            if self.config['auto_open_browser']:
                webbrowser.open(f"http://{frontend_config['host']}:{frontend_config['port']}")
                
        except Exception as e:
            logging.error(f"Frontend startup error: {str(e)}")
            raise
            
    def start(self):
        """Start the entire application"""
        try:
            logging.info("Starting Science Platform...")
            self.is_running = True
            
            # Start backend
            self.start_backend()
            
            # Start frontend
            self.start_frontend()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
            
            # Start background tasks
            asyncio.get_event_loop().create_task(self.run_background_tasks())
            
            logging.info("Science Platform started successfully")
            
            # Keep the main process running
            while self.is_running:
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
                
        except Exception as e:
            logging.error(f"Application startup error: {str(e)}")
            self.shutdown()
            
    def shutdown(self):
        """Shutdown the application"""
        logging.info("Shutting down Science Platform...")
        self.is_running = False
        
        # Stop frontend
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
            
        # Stop backend
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.join()
            
        logging.info("Science Platform shutdown complete")
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown()
        
    async def run_background_tasks(self):
        """Run periodic background tasks"""
        while self.is_running:
            try:
                # Check for file changes
                changes = self.change_tracker.detect_changes(Path(self.config['data_dir']))
                if any(changes.values()):
                    logging.info(f"Detected file changes: {changes}")
                    # Submit tasks for processing changes
                    await self.handle_file_changes(changes)
                    
                # Process task queue
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Background task error: {str(e)}")
                await asyncio.sleep(10)  # Wait longer on error
                
    async def handle_file_changes(self, changes: Dict[str, List[Dict[str, Any]]]):
        """Handle detected file changes"""
        try:
            # Handle modified files
            for change in changes['modified']:
                await self.task_manager.submit_task(
                    name="process_modified_file",
                    func=self.process_modified_file,
                    args=(change['path'], change['new_hash']),
                    priority=TaskPriority.HIGH
                )
                
            # Handle new files
            for change in changes['new']:
                await self.task_manager.submit_task(
                    name="process_new_file",
                    func=self.process_new_file,
                    args=(change['path'], change['hash']),
                    priority=TaskPriority.MEDIUM
                )
                
        except Exception as e:
            logging.error(f"File change handling error: {str(e)}")
            raise
            
    async def process_modified_file(self, file_path: str, file_hash: str):
        """Process a modified file"""
        # Implement file processing logic
        pass
        
    async def process_new_file(self, file_path: str, file_hash: str):
        """Process a new file"""
        # Implement file processing logic
        pass

def main():
    """Main entry point for the application"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    app_manager = ApplicationManager(config_path)
    app_manager.start()

if __name__ == "__main__":
    main() 