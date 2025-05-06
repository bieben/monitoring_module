"""
Factory class for creating result exporters
"""

from typing import Dict, Optional
from .redis_exporter import RedisExporter
from ..config import CACHE_CONFIG

class ExporterFactory:
    """Factory class for creating result exporters"""
    
    @staticmethod
    def create_exporter(exporter_type: str, config: Optional[Dict] = None) -> object:
        """
        Create an exporter instance based on type
        
        Args:
            exporter_type: Type of exporter ('redis')
            config: Optional configuration dictionary
            
        Returns:
            Exporter instance
        """
        if exporter_type.lower() == 'redis':
            return RedisExporter(config=config)
        else:
            raise ValueError(f"Unsupported exporter type: {exporter_type}")
    
    @staticmethod
    def get_default_config(exporter_type: str) -> Dict:
        """
        Get default configuration for an exporter type
        
        Args:
            exporter_type: Type of exporter ('redis')
            
        Returns:
            Configuration dictionary
        """
        if exporter_type.lower() == 'redis':
            return CACHE_CONFIG
        else:
            raise ValueError(f"Unsupported exporter type: {exporter_type}") 