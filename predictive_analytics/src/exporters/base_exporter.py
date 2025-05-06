"""
Base class for result exporters
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class BaseExporter(ABC):
    """Abstract base class for result exporters"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base exporter
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            raise ValueError("Missing required configuration")
        self.config = config
    
    @abstractmethod
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data
        
        Args:
            data: Data to export
            
        Returns:
            True if export successful
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Exporter must implement export method")
    
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Exporter must implement _validate_config method")
    
    def _format_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for export
        
        Args:
            data: Data to format
            
        Returns:
            Formatted data
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Exporter must implement _format_data method")
    
    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """
        Serialize data for storage
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        try:
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error serializing data: {str(e)}")
            raise RuntimeError(f"Failed to serialize data: {str(e)}")
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate data format
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        required_keys = {'predictions', 'optimization'}
        
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary")
            return False
            
        if not all(key in data for key in required_keys):
            logger.error(f"Missing required keys. Required: {required_keys}")
            return False
            
        # Validate predictions
        predictions = data.get('predictions')
        if not isinstance(predictions, pd.DataFrame):
            logger.error("Predictions must be a pandas DataFrame")
            return False
            
        # Validate optimization results
        optimization = data.get('optimization')
        if not isinstance(optimization, dict):
            logger.error("Optimization results must be a dictionary")
            return False
            
        return True
    
    def _format_data(self, data: Dict[str, Any]) -> Dict:
        """
        Format data for export
        
        Args:
            data: Raw data
            
        Returns:
            Formatted data dictionary
        """
        try:
            predictions_df = data['predictions']
            optimization_dict = data['optimization']
            
            # Convert timestamps to ISO format
            predictions_dict = predictions_df.to_dict(orient='records')
            for record in predictions_dict:
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp'].isoformat()
            
            formatted_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'predictions': predictions_dict,
                'optimization': optimization_dict
            }
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting data: {str(e)}")
            raise
    
    def _serialize_data(self, data: Dict) -> str:
        """
        Serialize data to JSON string
        
        Args:
            data: Data dictionary
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error serializing data: {str(e)}")
            raise 