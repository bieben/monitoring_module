"""
Factory class for creating resource optimizers
"""

from typing import Dict, Optional
from .pulp_optimizer import PuLPOptimizer
from ..config import OPTIMIZATION_CONFIG

class OptimizerFactory:
    """Factory class for creating resource optimizers"""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, config: Optional[Dict] = None) -> object:
        """
        Create an optimizer instance based on type
        
        Args:
            optimizer_type: Type of optimizer ('pulp')
            config: Optional configuration dictionary
            
        Returns:
            Optimizer instance
        """
        if optimizer_type.lower() == 'pulp':
            return PuLPOptimizer(config=config)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_default_config(optimizer_type: str) -> Dict:
        """
        Get default configuration for an optimizer type
        
        Args:
            optimizer_type: Type of optimizer ('pulp')
            
        Returns:
            Configuration dictionary
        """
        if optimizer_type.lower() == 'pulp':
            return OPTIMIZATION_CONFIG
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}") 