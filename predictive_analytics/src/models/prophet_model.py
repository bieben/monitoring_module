"""
Prophet model implementation
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .base_model import BaseModel
from ..config import PROPHET_CONFIG
import joblib
import os

logger = logging.getLogger(__name__)

class ProphetModel(BaseModel):
    """Prophet model for time series prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Prophet model
        
        Args:
            config: Prophet configuration dictionary
        """
        super().__init__(config)
        
        # 验证配置
        self._validate_config()
        
        # 设置模型参数
        self.model_params = {
            'changepoint_prior_scale': self.config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': self.config.get('seasonality_prior_scale', 10.0),
            'holidays_prior_scale': self.config.get('holidays_prior_scale', 10.0),
            'seasonality_mode': self.config.get('seasonality_mode', 'additive'),
            'yearly_seasonality': self.config.get('yearly_seasonality', False),
            'weekly_seasonality': self.config.get('weekly_seasonality', False),
            'daily_seasonality': self.config.get('daily_seasonality', True)
        }
        
        # 初始化模型
        self.models = {}
        self.metrics = ['cpu_usage', 'memory_usage', 'network_io', 'latency']
        self.data_validation = self.config.get('data_validation', {})
        self.is_trained = False
    
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {'data_validation'}
        if not all(field in self.config for field in required_fields):
            raise ValueError("Missing required configuration fields")
            
        if 'value_range' not in self.config['data_validation']:
            raise ValueError("Missing value range in data validation configuration")
            
        return True
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train Prophet models for each metric
        
        Args:
            data: Training data DataFrame
        """
        if not self._validate_data(data):
            raise ValueError("Invalid input data")
            
        data = self._prepare_data(data)
        
        try:
            for metric in self.metrics:
                # 准备Prophet数据
                df = pd.DataFrame({
                    'ds': pd.to_datetime(data['timestamp']),
                    'y': data[metric]
                })
                
                # 验证数据范围
                value_range = self.data_validation['value_range'].get(metric)
                if value_range:
                    if not df['y'].between(value_range[0], value_range[1]).all():
                        raise ValueError(f"{metric} values out of valid range: {value_range}")
                
                # 初始化并训练模型
                model = Prophet(
                    **self.model_params
                )
                
                # 训练模型
                model.fit(df)
                
                # 保存模型
                self.models[metric] = model
                
            self.is_trained = True
            logger.info("Prophet models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Prophet models: {str(e)}")
            raise
    
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Make predictions using trained Prophet models
        
        Args:
            horizon: Number of future time points to predict
            
        Returns:
            DataFrame containing predictions
            
        Raises:
            ValueError: If models are not trained or horizon is invalid
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
            
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
            
        try:
            # 创建未来日期范围
            future_dates = pd.date_range(
                start=datetime.now(),
                periods=horizon,
                freq='5min'
            )
            
            predictions = pd.DataFrame({'timestamp': future_dates})
            
            # 对每个指标进行预测
            for metric in self.metrics:
                future = pd.DataFrame({'ds': future_dates})
                forecast = self.models[metric].predict(future)
                
                # 确保预测值在有效范围内
                value_range = self.data_validation['value_range'].get(metric)
                if value_range:
                    forecast['yhat'] = forecast['yhat'].clip(value_range[0], value_range[1])
                    if 'yhat_lower' in forecast.columns:
                        forecast['yhat_lower'] = forecast['yhat_lower'].clip(value_range[0], value_range[1])
                    if 'yhat_upper' in forecast.columns:
                        forecast['yhat_upper'] = forecast['yhat_upper'].clip(value_range[0], value_range[1])
                
                # 重命名预测列
                predictions[metric] = forecast['yhat']
                
                # 添加预测区间
                if self.model_params.get('return_intervals', True):
                    predictions[f"{metric}_lower"] = forecast['yhat_lower']
                    predictions[f"{metric}_upper"] = forecast['yhat_upper']
            
            # 统一时间戳格式
            predictions['timestamp'] = predictions['timestamp'].dt.strftime(
                self.data_validation['timestamp_format']
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid
        """
        required_columns = {'timestamp'}.union(set(self.metrics))
        
        if not isinstance(data, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return False
            
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
            
        if data.empty:
            logger.error("Input DataFrame is empty")
            return False
            
        # 验证时间戳
        try:
            pd.to_datetime(data['timestamp'])
        except:
            logger.error("Column timestamp must be datetime")
            return False
                
        # 验证指标数据类型和范围
        for metric in self.metrics:
            if not pd.api.types.is_numeric_dtype(data[metric]):
                logger.error(f"Column {metric} must be numeric")
                return False
                
            value_range = self.data_validation['value_range'].get(metric)
            if value_range and not data[metric].between(value_range[0], value_range[1]).all():
                logger.error(f"{metric} values out of valid range: {value_range}")
                return False
                
        return True
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        df = data.copy()
        
        # 确保时间戳是datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 设置时间戳为索引
        df = df.set_index('timestamp')
        
        # 按时间戳排序
        df = df.sort_index()
        
        # 移除重复的时间戳
        df = df[~df.index.duplicated(keep='first')]
        
        # 处理缺失值
        for metric in self.metrics:
            df[metric] = df[metric].interpolate(method='linear')
            
            # 确保插值后的数据在有效范围内
            value_range = self.data_validation['value_range'].get(metric)
            if value_range:
                df[metric] = df[metric].clip(value_range[0], value_range[1])
        
        # 重置索引，保持timestamp作为列
        df = df.reset_index()
        
        # 统一时间戳格式
        df['timestamp'] = df['timestamp'].dt.strftime(self.data_validation['timestamp_format'])
        
        return df
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: Path to save model to
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'config': self.config,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
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
            model_data = joblib.load(path)
            self.models = model_data['models']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def save_model(self) -> None:
        """Save Prophet models to cache directory"""
        if not self.is_trained:
            logger.warning("Cannot save untrained models")
            return
            
        try:
            cache_dir = self.config.get('cache_dir', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            model_path = f"{cache_dir}/prophet_model.joblib"
            
            model_data = {
                'models': self.models,
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Prophet models saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving Prophet models: {str(e)}")
            raise
    
    def load_model(self) -> bool:
        """
        Load Prophet models from cache directory
        
        Returns:
            True if models were loaded successfully
        """
        try:
            cache_dir = self.config.get('cache_dir', 'cache')
            model_path = f"{cache_dir}/prophet_model.joblib"
            
            if not os.path.exists(model_path):
                logger.info(f"No cached model found at {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            self.models = model_data['models']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            logger.info("Prophet models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Prophet models: {str(e)}")
            return False 