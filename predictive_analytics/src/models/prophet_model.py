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
        self.metrics = ['requests_total', 'latency_avg', 'latency_p95', 'latency_p99']
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
        processed_data = self._validate_data(data)
        if processed_data is None:
            raise ValueError("Invalid input data")
            
        data = self._prepare_data(processed_data)
        
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
                    # 检查是否有超出范围的值
                    out_of_range = df[(df['y'] < value_range[0]) | (df['y'] > value_range[1])]
                    if not out_of_range.empty:
                        logger.warning(f"Found {len(out_of_range)} values out of range for metric {metric}. Clipping to range {value_range}")
                        # 修剪超出范围的值
                        df['y'] = df['y'].clip(value_range[0], value_range[1])
                
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
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data format and correct if needed
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed and validated DataFrame or None if invalid
        """
        # 检查数据格式
        if not isinstance(data, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return None
            
        if data.empty:
            logger.error("Input DataFrame is empty")
            return None
        
        # 从原始数据提取所需指标
        processed_df = self._extract_metrics_from_raw_data(data)
        
        # 验证处理后的数据是否包含所需的列
        required_columns = {'timestamp'}.union(set(self.metrics))
        if not all(col in processed_df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return None
        
        # 验证时间戳
        try:
            pd.to_datetime(processed_df['timestamp'])
        except:
            logger.error("Column timestamp must be datetime")
            return None
                
        # 验证指标数据类型和范围
        for metric in self.metrics:
            if not pd.api.types.is_numeric_dtype(processed_df[metric]):
                logger.error(f"Column {metric} must be numeric")
                return None
                
            value_range = self.data_validation['value_range'].get(metric)
            if value_range:
                # 检查是否有超出范围的值
                out_of_range = processed_df[(processed_df[metric] < value_range[0]) | (processed_df[metric] > value_range[1])]
                if not out_of_range.empty:
                    logger.warning(f"Found {len(out_of_range)} values out of range for metric {metric}. Clipping to range {value_range}")
                    # 修剪超出范围的值
                    processed_df[metric] = processed_df[metric].clip(value_range[0], value_range[1])
                
        return processed_df
    
    def _extract_metrics_from_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        从原始数据中提取每个指标，并转换为适合模型的格式
        
        Args:
            data: 原始数据，包含metric_name, value, timestamp, model_id列
            
        Returns:
            DataFrame，每个指标作为一列，包含timestamp和所有指标值
        """
        if 'metric_name' not in data.columns or 'value' not in data.columns or 'timestamp' not in data.columns:
            logger.error("Raw data must contain metric_name, value, and timestamp columns")
            return pd.DataFrame()
        
        # 确保timestamp是datetime类型
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # 创建结果DataFrame
        result_df = pd.DataFrame()
        
        # 按模型ID分组处理（如果存在)
        if 'model_id' in data.columns:
            # 为简化起见，我们只使用第一个模型ID的数据
            first_model_id = data['model_id'].iloc[0]
            data = data[data['model_id'] == first_model_id]
        
        # 获取所有不同的时间戳
        unique_timestamps = data['timestamp'].unique()
        result_df['timestamp'] = unique_timestamps
        
        # 对于每个指标，提取数据并添加到结果DataFrame中
        for metric in self.metrics:
            metric_data = data[data['metric_name'] == metric]
            if not metric_data.empty:
                # 为每个时间戳取一个值（可能需要更复杂的聚合）
                metric_df = metric_data.groupby('timestamp')['value'].mean().reset_index()
                metric_df.columns = ['timestamp', metric]
                
                # 合并到结果DataFrame
                result_df = pd.merge(result_df, metric_df, on='timestamp', how='left')
        
        # 确保所有指标列都存在，如果不存在则创建并填充0
        for metric in self.metrics:
            if metric not in result_df.columns:
                result_df[metric] = 0.0
        
        # 按时间戳排序
        result_df = result_df.sort_values('timestamp')
        
        # 处理缺失值
        for metric in self.metrics:
            result_df[metric] = result_df[metric].interpolate(method='linear')
        
        return result_df
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        # 从原始数据提取指标
        if 'metric_name' in data.columns and 'value' in data.columns:
            df = self._extract_metrics_from_raw_data(data)
        else:
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
            if metric in df.columns:
                df[metric] = df[metric].interpolate(method='linear')
                
                # 确保插值后的数据在有效范围内
                value_range = self.data_validation['value_range'].get(metric)
                if value_range:
                    df[metric] = df[metric].clip(value_range[0], value_range[1])
        
        # 重置索引，恢复timestamp列
        df = df.reset_index()
        
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