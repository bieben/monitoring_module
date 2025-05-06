"""
Base class for resource optimizers
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseOptimizer(ABC):
    """Abstract base class for resource optimizers"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base optimizer
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            raise ValueError("Missing required configuration")
        self.config = config
        
    def optimize(self, predictions: pd.DataFrame, constraints: Dict) -> Dict:
        """
        Optimize resource allocation
        
        Args:
            predictions: DataFrame containing predictions
            constraints: Dictionary containing constraints
            
        Returns:
            Dictionary containing optimization results
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Optimizer must implement optimize method")
        
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Optimizer must implement _validate_config method")
    
    def _validate_weights(self, weights: Dict) -> bool:
        """
        Validate optimization weights
        
        Args:
            weights: Weight dictionary
            
        Returns:
            True if weights are valid
        """
        required_weights = {'cpu', 'memory', 'network'}
        
        if not isinstance(weights, dict):
            logger.error("Weights must be a dictionary")
            return False
            
        if not all(key in weights for key in required_weights):
            logger.error(f"Missing required weights. Required: {required_weights}")
            return False
            
        # 验证权重值
        total_weight = 0
        for key, value in weights.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                logger.error(f"Invalid weight value for {key}: {value}")
                return False
            total_weight += value
            
        # 验证权重和为1
        if not abs(total_weight - 1.0) < 1e-6:
            logger.error(f"Weights must sum to 1.0, got {total_weight}")
            return False
            
        return True
    
    def _validate_predictions(self, predictions: pd.DataFrame) -> bool:
        """
        Validate prediction data format
        
        Args:
            predictions: Prediction DataFrame
            
        Returns:
            True if predictions are valid
        """
        required_columns = {'timestamp', 'cpu_usage', 'memory_usage', 'network_io', 'latency'}
        
        if not isinstance(predictions, pd.DataFrame):
            logger.error("Predictions must be a pandas DataFrame")
            return False
            
        if not all(col in predictions.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
            
        if predictions.empty:
            logger.error("Predictions DataFrame is empty")
            return False
            
        # 验证数据类型
        numeric_columns = {'cpu_usage', 'memory_usage', 'network_io', 'latency'}
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(predictions[col]):
                logger.error(f"Column {col} must be numeric")
                return False
                
        # 验证时间戳
        if not pd.api.types.is_datetime64_any_dtype(predictions['timestamp']):
            try:
                pd.to_datetime(predictions['timestamp'])
            except:
                logger.error("Column timestamp must be datetime")
                return False
                
        return True
    
    def _validate_constraints(self, constraints: Dict) -> bool:
        """
        Validate constraint format
        
        Args:
            constraints: Constraint dictionary
            
        Returns:
            True if constraints are valid
        """
        required_constraints = {'max_cpu', 'max_memory', 'max_network', 'max_latency'}
        
        if not isinstance(constraints, dict):
            logger.error("Constraints must be a dictionary")
            return False
            
        if not all(key in constraints for key in required_constraints):
            logger.error(f"Missing required constraints. Required: {required_constraints}")
            return False
            
        # 验证约束值
        for key, value in constraints.items():
            if not isinstance(value, (int, float)) or value <= 0:
                logger.error(f"Invalid constraint value for {key}: {value}")
                return False
                
        return True 