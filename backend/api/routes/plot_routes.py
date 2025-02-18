from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

router = APIRouter()

@router.post("/plot/generate")
async def generate_plot(plot_data: Dict[str, Any]):
    """Generate plot based on experiment data"""
    try:
        plot = await db_manager.generate_plot(
            plot_type=plot_data['type'],
            data=plot_data['data'],
            params=plot_data.get('params', {})
        )
        return plot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plot/{plot_id}")
async def get_plot(plot_id: str):
    """Retrieve cached plot data"""
    try:
        plot_data = await db_manager.get_cached_plot(plot_id)
        if not plot_data:
            raise HTTPException(status_code=404, detail="Plot not found")
        return plot_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 