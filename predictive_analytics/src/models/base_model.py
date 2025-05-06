"""
Base class for prediction models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for prediction models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base model
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            raise ValueError("Missing required configuration")
        self.config = config
        
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """
        Train model
        
        Args:
            data: Training data
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Model must implement train method")
    
    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            horizon: Number of time steps to predict
            
        Returns:
            DataFrame containing predictions
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: Path to save model
        """
        try:
            # 创建目录
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            joblib.dump(self, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self, path: str) -> None:
        """
        Load model from file
        
        Args:
            path: Path to load model from
        """
        try:
            # 加载模型
            loaded_model = joblib.load(path)
            
            # 复制属性
            self.__dict__.update(loaded_model.__dict__)
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Empty data")
            
        required_columns = {'timestamp'} | set(self.metrics)
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return True
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for model training/prediction
        
        Args:
            data: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        df = data.copy()
        
        # 确保timestamp列是datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 设置timestamp为索引
        df = df.set_index('timestamp')
        
        # 按时间戳排序
        df = df.sort_index()
        
        # 移除重复的时间戳
        df = df[~df.index.duplicated(keep='first')]
        
        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].interpolate(method='time')
            
        # 重置索引，保持timestamp作为列
        df = df.reset_index()
        
        return df 