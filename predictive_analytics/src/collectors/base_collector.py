from typing import Optional, Dict
import pandas as pd
from datetime import datetime

class BaseCollector:
    """Base class for all collectors"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base collector
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            raise ValueError("Missing required configuration")
        self.config = config
        
    def collect_metrics(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Collect metrics
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            DataFrame containing collected metrics
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Collector must implement collect_metrics method")
        
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Collector must implement _validate_config method") 