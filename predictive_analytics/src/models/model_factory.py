"""
Factory class for creating prediction models
"""

from typing import Dict, Optional
from .prophet_model import ProphetModel
from .sklearn_model import SklearnModel
from ..config import PROPHET_CONFIG, SKLEARN_CONFIG

class ModelFactory:
    """Factory class for creating prediction models"""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict] = None) -> object:
        """
        Create a model instance based on type
        
        Args:
            model_type: Type of model ('prophet' or 'sklearn')
            config: Optional configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is invalid
        """
        if model_type.lower() == 'prophet':
            return ProphetModel(config=config)
        elif model_type.lower() == 'sklearn':
            return SklearnModel(config=config)
        else:
            raise ValueError("Invalid model type")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict:
        """
        Get default configuration for a model type
        
        Args:
            model_type: Type of model ('prophet' or 'sklearn')
            
        Returns:
            Configuration dictionary
        """
        if model_type.lower() == 'prophet':
            return PROPHET_CONFIG
        elif model_type.lower() == 'sklearn':
            return SKLEARN_CONFIG
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 