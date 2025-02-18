from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any
import logging
import json
import time
from datetime import datetime
import uuid
from pathlib import Path

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self, app, log_dir: Path = Path("logs")):
        super().__init__(app)
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("api")
        self.logger.setLevel(logging.INFO)
        
        # File handler for all requests
        all_handler = logging.FileHandler(
            self.log_dir / "all_requests.log"
        )
        all_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(all_handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            self.log_dir / "error.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(error_handler)
        
    async def dispatch(self, 
                      request: Request, 
                      call_next: Callable) -> Response:
        """Log request and response details"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Prepare request logging
        request_log = {
            "id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "headers": dict(request.headers)
        }
        
        # Log request body for non-GET requests
        if request.method != "GET":
            try:
                body = await request.json()
                request_log["body"] = self._sanitize_data(body)
            except:
                request_log["body"] = "Could not parse body"
        
        response = None
        try:
            response = await call_next(request)
            
            # Log response
            response_log = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": time.time() - start_time,
                "headers": dict(response.headers)
            }
            
            # Log complete request-response cycle
            self.logger.info(json.dumps({
                "request": request_log,
                "response": response_log
            }))
            
            return response
            
        except Exception as e:
            # Log error
            error_log = {
                "request_id": request_id,
                "error": str(e),
                "traceback": logging.traceback.format_exc()
            }
            
            self.logger.error(json.dumps({
                "request": request_log,
                "error": error_log
            }))
            
            raise
            
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from logged data"""
        sensitive_fields = {'password', 'token', 'secret', 'key'}
        
        if isinstance(data, dict):
            return {
                k: '***' if k.lower() in sensitive_fields 
                else self._sanitize_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data

class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging performance metrics"""
    
    def __init__(self, app, threshold_ms: float = 500):
        super().__init__(app)
        self.threshold_ms = threshold_ms
        self.logger = logging.getLogger("performance")
        
        # Configure performance logging
        perf_handler = logging.FileHandler("logs/performance.log")
        perf_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(message)s'
            )
        )
        self.logger.addHandler(perf_handler)
        
    async def dispatch(self, 
                      request: Request, 
                      call_next: Callable) -> Response:
        """Log request performance metrics"""
        start_time = time.time()
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        
        if duration_ms > self.threshold_ms:
            self.logger.warning(
                f"Slow request: {request.method} {request.url} "
                f"took {duration_ms:.2f}ms"
            )
            
        # Log all performance metrics
        self.logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "duration_ms": duration_ms,
            "status_code": response.status_code
        }))
        
        return response
