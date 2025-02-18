from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.middleware.logging import RequestLoggingMiddleware, PerformanceLoggingMiddleware
from api.routes import chat_routes, search_routes, experiment_routes, graph_routes, plot_routes
from api.middleware.auth import auth_handler
import logging
from pathlib import Path
from app_manager import ApplicationManager

# Configure base logging
logging.basicConfig(level=logging.INFO)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Initialize application manager
app_manager = ApplicationManager()

# Initialize FastAPI app
app = FastAPI(
    title="Science Platform API",
    description="API for scientific experiment management and analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PerformanceLoggingMiddleware)

# Include routers
app.include_router(chat_routes.router, prefix="/chat", tags=["chat"])
app.include_router(search_routes.router, prefix="/search", tags=["search"])
app.include_router(experiment_routes.router, prefix="/experiments", tags=["experiments"])
app.include_router(graph_routes.router, prefix="/knowledge", tags=["knowledge"])
app.include_router(plot_routes.router, prefix="/plot", tags=["plot"])

# Make app_manager available to routes
@app.middleware("http")
async def add_app_manager(request: Request, call_next):
    request.state.app_manager = app_manager
    response = await call_next(request)
    return response

# Example of using pipeline managers in routes
@app.post("/experiments/{experiment_type}")
async def create_experiment(
    experiment_type: str,
    input_data: dict,
    request: Request
):
    try:
        result = await request.state.app_manager.create_experiment(
            experiment_type=experiment_type,
            input_data=input_data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Protected route example
@app.get("/protected")
async def protected_route(request: Request):
    user = await auth_handler.get_current_user(request)
    return {"message": f"Hello {user['username']}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 