from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import asyncio
import logging
from datetime import datetime
import uuid
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    """Represents a task in the system"""
    id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    dependencies: List[str]
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    machine_id: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'machine_id': self.machine_id,
            'retries': self.retries,
            'max_retries': self.max_retries
        }

class TaskManager:
    """Manages distributed task execution and monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.machine_id = self.config.get('machine_id', self._generate_machine_id())
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_handlers: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_dependencies = defaultdict(set)
        self.load_state()
        self._start_monitor()
        
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
        
    async def submit_task(self,
                         name: str,
                         func: Callable,
                         args: tuple = (),
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         dependencies: List[str] = None) -> str:
        """Submit a new task"""
        try:
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                name=name,
                func=func,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                dependencies=dependencies or [],
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow().timestamp()
            )
            
            self.tasks[task_id] = task
            
            # Add to dependency tracking
            for dep_id in task.dependencies:
                self.task_dependencies[dep_id].add(task_id)
            
            # Add to queue if no dependencies
            if not task.dependencies:
                self.task_queue.put((-priority.value, task_id))
            
            self.save_state()
            return task_id
            
        except Exception as e:
            logging.error(f"Task submission error: {str(e)}")
            raise
            
    async def execute_task(self, task_id: str):
        """Execute a task"""
        try:
            task = self.tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow().timestamp()
            task.machine_id = self.machine_id
            
            try:
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, task.func, *task.args, **task.kwargs
                    )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow().timestamp()
                
                # Process dependent tasks
                self._process_dependent_tasks(task_id)
                
            except Exception as e:
                task.error = str(e)
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.status = TaskStatus.PENDING
                    self.task_queue.put((-task.priority.value, task_id))
                else:
                    task.status = TaskStatus.FAILED
                
            self.save_state()
            
        except Exception as e:
            logging.error(f"Task execution error: {str(e)}")
            raise
            
    def cancel_task(self, task_id: str):
        """Cancel a task"""
        try:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
                    self.save_state()
                    
        except Exception as e:
            logging.error(f"Task cancellation error: {str(e)}")
            raise
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and details"""
        try:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
            return None
            
        except Exception as e:
            logging.error(f"Task status retrieval error: {str(e)}")
            raise
            
    def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on the completed task"""
        dependent_tasks = self.task_dependencies[completed_task_id]
        for task_id in dependent_tasks:
            task = self.tasks[task_id]
            task.dependencies.remove(completed_task_id)
            if not task.dependencies:
                self.task_queue.put((-task.priority.value, task_id))
                
    def _start_monitor(self):
        """Start task monitoring thread"""
        def monitor():
            while True:
                try:
                    _, task_id = self.task_queue.get()
                    task = self.tasks[task_id]
                    if task.status == TaskStatus.PENDING:
                        asyncio.run(self.execute_task(task_id))
                except Exception as e:
                    logging.error(f"Task monitor error: {str(e)}")
                    
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
    def _generate_machine_id(self) -> str:
        """Generate unique machine identifier"""
        import socket
        import hashlib
        return hashlib.md5(socket.gethostname().encode()).hexdigest()
        
    def save_state(self):
        """Save task manager state"""
        state_path = Path(self.config.get('state_path', 'data/tasks/state.json'))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'tasks': {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
            'dependencies': {
                task_id: list(deps)
                for task_id, deps in self.task_dependencies.items()
            }
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f)
            
    def load_state(self):
        """Load task manager state"""
        state_path = Path(self.config.get('state_path', 'data/tasks/state.json'))
        
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Reconstruct tasks and dependencies
            for task_id, task_data in state['tasks'].items():
                task_data['priority'] = TaskPriority(task_data['priority'])
                task_data['status'] = TaskStatus(task_data['status'])
                self.tasks[task_id] = Task(**task_data)
                
            self.task_dependencies = defaultdict(set)
            for task_id, deps in state['dependencies'].items():
                self.task_dependencies[task_id] = set(deps) 