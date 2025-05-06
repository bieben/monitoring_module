"""
Scikit-learn model implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from .base_model import BaseModel
from ..config import SKLEARN_CONFIG
import joblib
import os

logger = logging.getLogger(__name__)

class SklearnModel(BaseModel):
    """Scikit-learn model for time series forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Scikit-learn model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # 验证配置
        self._validate_config()
        
        # 初始化模型参数
        self.models = {}  # 每个指标一个模型
        self.scalers = {}  # 每个指标一个标准化器
        self.metrics = ['cpu_usage', 'memory_usage', 'network_io', 'latency']
        self.lookback = self.config.get('lookback', 12)  # 用于预测的历史点数
        self.is_trained = False
        
        # 添加数据验证配置
        self.data_validation = self.config.get('data_validation', {})
    
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {'data_validation', 'model_params'}
        if not all(field in self.config for field in required_fields):
            raise ValueError("Missing required configuration fields")
            
        if 'value_range' not in self.config['data_validation']:
            raise ValueError("Missing value range in data validation configuration")
            
        return True
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train Scikit-learn models for each metric
        
        Args:
            data: Training data DataFrame
        """
        if not self._validate_data(data):
            raise ValueError("Invalid input data")
        
        try:
            # 准备特征
            prepared_data = self._prepare_features(data)
            
            for metric in self.metrics:
                # 准备训练数据
                X, y = self._prepare_sequences(prepared_data, metric)
                
                # 标准化特征
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[metric] = scaler
                
                # 初始化并训练模型
                model = RandomForestRegressor(**self.config['model_params'])
                model.fit(X_scaled, y)
                
                self.models[metric] = model
            
            self.is_trained = True
            logger.info("Scikit-learn models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Scikit-learn models: {str(e)}")
            raise
    
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Make predictions using trained Scikit-learn models
        
        Args:
            horizon: Number of minutes to forecast
            
        Returns:
            DataFrame with predictions for all metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            future_dates = pd.date_range(
                start=datetime.now(),
                periods=horizon,
                freq='5min'
            )
            
            # 准备时间特征
            future_features = self._prepare_time_features(future_dates)
            predictions = {}
            
            for metric in self.metrics:
                # 获取最近的序列
                last_sequence = self._get_last_sequence(metric)
                
                # 预测
                metric_predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(horizon):
                    # 准备特征
                    time_features = future_features[i:i+1]
                    sequence_features = current_sequence.reshape(1, -1)
                    
                    # 合并特征
                    features = np.hstack([sequence_features, time_features])
                    
                    # 标准化
                    features_scaled = self.scalers[metric].transform(features)
                    
                    # 预测
                    pred = self.models[metric].predict(features_scaled)[0]
                    
                    # 确保预测值在有效范围内
                    value_range = self.data_validation['value_range'].get(metric)
                    if value_range:
                        pred = np.clip(pred, value_range[0], value_range[1])
                    
                    metric_predictions.append(pred)
                    
                    # 更新序列
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = pred
                
                predictions[metric] = metric_predictions
            
            # 创建结果DataFrame
            result = pd.DataFrame({
                'timestamp': future_dates.strftime(self.data_validation['timestamp_format']),
                **predictions
            })
            
            # 确保所有必需的列都存在
            for metric in ['cpu_usage', 'memory_usage', 'network_io', 'latency']:
                if metric not in result.columns:
                    result[metric] = 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        df = data.copy()
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 添加时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # 添加滞后特征
        for metric in self.metrics:
            for lag in range(1, self.lookback + 1):
                df[f"{metric}_lag_{lag}"] = df[metric].shift(lag)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        return df
    
    def _prepare_time_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Prepare time features for prediction
        
        Args:
            dates: DatetimeIndex of future dates
            
        Returns:
            Array of time features
        """
        features = np.column_stack([
            dates.hour,
            dates.dayofweek,
            dates.month,
            dates.day,
            dates.isocalendar().week
        ])
        return features
    
    def _prepare_sequences(self, data: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training
        
        Args:
            data: Prepared DataFrame
            metric: Target metric
            
        Returns:
            Tuple of (X, y) arrays
        """
        # 准备特征列
        feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_month', 'week_of_year'
        ]
        feature_cols.extend([f"{metric}_lag_{i}" for i in range(1, self.lookback + 1)])
        
        X = data[feature_cols].values
        y = data[metric].values
        
        return X, y
    
    def _get_last_sequence(self, metric: str) -> np.ndarray:
        """
        Get the last sequence of values for a metric
        
        Args:
            metric: Metric name
            
        Returns:
            Array of last lookback values
        """
        # 这里应该实现获取实际的最后序列
        # 暂时返回零序列作为占位符
        return np.zeros(self.lookback)
    
    def save_model(self) -> None:
        """Save all Scikit-learn models and scalers"""
        if not self.is_trained:
            logger.warning("Cannot save untrained models")
            return
            
        try:
            for metric in self.metrics:
                # 保存模型
                model_path = f"{self.cache_dir}/{self.model_name}_{metric}_model.joblib"
                joblib.dump(self.models[metric], model_path)
                
                # 保存标准化器
                scaler_path = f"{self.cache_dir}/{self.model_name}_{metric}_scaler.joblib"
                joblib.dump(self.scalers[metric], scaler_path)
                
            logger.info("Scikit-learn models and scalers saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_model(self) -> bool:
        """
        Load all Scikit-learn models and scalers
        
        Returns:
            True if all models were loaded successfully
        """
        try:
            for metric in self.metrics:
                # 加载模型
                model_path = f"{self.cache_dir}/{self.model_name}_{metric}_model.joblib"
                scaler_path = f"{self.cache_dir}/{self.model_name}_{metric}_scaler.joblib"
                
                if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                    return False
                
                self.models[metric] = joblib.load(model_path)
                self.scalers[metric] = joblib.load(scaler_path)
            
            self.is_trained = True
            logger.info("Scikit-learn models and scalers loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: Path to save model to
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'lookback': self.lookback
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
            self.scalers = model_data['scalers']
            self.lookback = model_data['lookback']
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") 