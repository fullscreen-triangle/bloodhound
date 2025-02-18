from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
from ...core.pipeline_manager import PipelineManager
from ...core.db_manager import DistributedDBManager

router = APIRouter()
pipeline_manager = PipelineManager()
db_manager = DistributedDBManager()

@router.post("/metabolomics")
async def run_metabolomics_analysis(input_data: Dict[str, Any]):
    """Run metabolomics analysis"""
    experiment_id = f"metabolomics_{int(datetime.now().timestamp())}"
    
    try:
        results = await pipeline_manager.metabolomics.run_analysis(
            input_data=input_data,
            experiment_id=experiment_id
        )
        
        await db_manager.store_experiment({
            'metadata': {
                'id': experiment_id,
                'type': 'metabolomics',
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        })
        
        return results
        
    except Exception as e:
        logging.error(f"Metabolomics analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/genomics")
async def run_genomics_analysis(input_data: Dict[str, Any]):
    """Run genomics analysis"""
    experiment_id = f"genomics_{int(datetime.now().timestamp())}"
    
    try:
        results = await pipeline_manager.genomics.run_analysis(
            input_data=input_data,
            experiment_id=experiment_id
        )
        
        await db_manager.store_experiment({
            'metadata': {
                'id': experiment_id,
                'type': 'genomics',
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        })
        
        return results
        
    except Exception as e:
        logging.error(f"Genomics analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Retrieve experiment details"""
    try:
        experiment = await db_manager.retrieve_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment
    except Exception as e:
        logging.error(f"Experiment retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 