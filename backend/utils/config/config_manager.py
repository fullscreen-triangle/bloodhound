from typing import Dict, Any, Optional, Tuple, List
import yaml
import json
from pathlib import Path
import logging

class ConfigManager:
    """Manages application configuration and experimental templates"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.config = self.load_main_config()
        self.templates = self.load_experimental_templates()
        
    def load_main_config(self) -> Dict[str, Any]:
        """Load main application configuration"""
        config_path = self.root_dir / "config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading main config: {str(e)}")
            return {}
            
    def load_experimental_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load experimental templates"""
        templates = {}
        template_dir = self.root_dir / "config" / "experimental_templates"
        
        try:
            for template_file in template_dir.glob("*.yaml"):
                with open(template_file, 'r') as f:
                    template_name = template_file.stem
                    templates[template_name] = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading templates: {str(e)}")
            
        return templates
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get specific experimental template"""
        return self.templates.get(template_name)
    
    def validate_experiment_config(self, 
                                 config: Dict[str, Any], 
                                 template_name: str) -> Tuple[bool, List[str]]:
        """Validate experiment configuration against template"""
        template = self.get_template(template_name)
        if not template:
            return False, [f"Template {template_name} not found"]
            
        errors = []
        # Implement validation logic based on template requirements
        return len(errors) == 0, errors 