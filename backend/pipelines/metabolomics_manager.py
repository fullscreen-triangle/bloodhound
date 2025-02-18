from typing import Dict, Any
import logging
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor
import os

from lavoisier.numeric import MSAnalysisPipeline
from lavoisier.visual import MSImageDatabase, MSVideoAnalyzer

class MetabolomicsPipelineManager:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.config = self._initialize_config()
        
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize configuration based on system resources"""
        specs = self.resource_manager.get_optimal_config()
        
        # Base configuration
        config = {
            'ms_parameters': {
                'n_workers': specs['num_workers'],
                'output_dir': str(Path('./output/metabolomics').absolute()),
                'intensity_threshold_ms1': 1000.0,
                'intensity_threshold_ms2': 100.0,
                'mz_tolerance': 0.01,
                'rt_tolerance': 0.5
            },
            'logging': {
                'file': str(Path('./logs/metabolomics.log').absolute()),
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'visualization': {
                'resolution': (1024, 1024),
                'feature_dimension': 128,
                'video_output_path': str(Path('./output/metabolomics/videos').absolute())
            }
        }
        
        # Adjust based on available resources
        if specs['use_gpu']:
            config['ms_parameters']['use_gpu'] = True
            config['ms_parameters']['gpu_memory_limit'] = specs['gpu_info'][0]['memory']
        
        return config
    
    async def run_analysis(self, 
                          input_data: Dict[str, Any],
                          experiment_id: str) -> Dict[str, Any]:
        """Run metabolomics analysis pipeline"""
        try:
            # Create experiment-specific directories
            exp_dir = Path(f'./data/experiments/{experiment_id}')
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save input files
            input_dir = exp_dir / 'input'
            input_dir.mkdir(exist_ok=True)
            for file_name, file_data in input_data['files'].items():
                file_path = input_dir / file_name
                with open(file_path, 'wb') as f:
                    f.write(file_data)
            
            # Update config with experiment-specific paths
            exp_config = self.config.copy()
            exp_config['ms_parameters']['output_dir'] = str(exp_dir / 'output')
            exp_config['logging']['file'] = str(exp_dir / 'logs' / 'analysis.log')
            exp_config['visualization']['video_output_path'] = str(exp_dir / 'videos')
            
            # Save experiment config
            config_path = exp_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(exp_config, f)
            
            # Initialize and run numerical analysis
            numerical_pipeline = MSAnalysisPipeline(str(config_path))
            numerical_results = numerical_pipeline.process_files(str(input_dir))
            
            # Run visual analysis if requested
            if input_data.get('generate_visualizations', True):
                visual_results = await self._run_visual_analysis(
                    numerical_results,
                    exp_dir,
                    exp_config
                )
            else:
                visual_results = None
            
            # Compile results
            results = {
                'experiment_id': experiment_id,
                'numerical_results': numerical_results,
                'visual_results': visual_results,
                'config': exp_config,
                'status': 'completed',
                'output_paths': {
                    'numerical': str(exp_dir / 'output'),
                    'visual': str(exp_dir / 'videos') if visual_results else None
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in metabolomics analysis: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _run_visual_analysis(self, 
                                 numerical_results: Dict[str, Any],
                                 exp_dir: Path,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Run visual analysis pipeline"""
        try:
            # Initialize image database
            db = MSImageDatabase(
                resolution=config['visualization']['resolution'],
                feature_dimension=config['visualization']['feature_dimension']
            )
            
            # Process spectra and add to database
            processed_spectra = []
            with ThreadPoolExecutor(max_workers=config['ms_parameters']['n_workers']) as executor:
                for result in numerical_results.values():
                    for spectrum in result['spectra']:
                        processed_spectra.append(
                            executor.submit(db.process_spectrum, spectrum)
                        )
            
            # Create video analysis
            video_analyzer = MSVideoAnalyzer()
            video_path = Path(config['visualization']['video_output_path'])
            video_path.mkdir(parents=True, exist_ok=True)
            
            video_data = [(s.mz_array, s.intensity_array) for s in processed_spectra]
            video_analyzer.extract_spectra_as_video(
                video_data,
                str(video_path / 'analysis.mp4')
            )
            
            return {
                'database_path': str(exp_dir / 'database'),
                'video_path': str(video_path / 'analysis.mp4'),
                'num_spectra_processed': len(processed_spectra)
            }
            
        except Exception as e:
            logging.error(f"Error in visual analysis: {str(e)}")
            raise 