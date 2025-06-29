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
import pytz
from pathlib import Path

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
    
    def train(self, data: pd.DataFrame):
        """
        Train Prophet models on the data
        
        Args:
            data: DataFrame containing metrics
        """
        logger.info(f"Training Prophet models on data with shape {data.shape}")
        
        # 获取数据中的唯一指标名称
        metrics = data['metric_name'].unique()
        logger.info(f"Metrics in training data: {metrics}")
        
        # 检查是否有真实资源指标
        real_metrics = [m for m in metrics if m in ['cpu_usage_real', 'memory_usage_real', 'network_io_real']]
        if real_metrics:
            logger.info(f"Found real system metrics in training data: {real_metrics}")
            
        # 对每个指标训练一个单独的模型
        for metric in metrics:
            # 过滤出当前指标的数据
            metric_data = data[data['metric_name'] == metric].copy()
            
            if metric_data.empty:
                logger.warning(f"No data for metric {metric}, skipping")
                continue
                
            # 确保数据按时间排序
            metric_data = metric_data.sort_values('timestamp')
            
            try:
                # 确保时间戳是datetime类型并处理时区
                metric_data['timestamp'] = pd.to_datetime(metric_data['timestamp'])
                
                # 如果时间戳没有时区信息，添加UTC时区
                if metric_data['timestamp'].dt.tz is None:
                    metric_data['timestamp'] = metric_data['timestamp'].dt.tz_localize('UTC')
                # 如果时间戳已有时区信息但不是UTC，转换为UTC
                elif str(metric_data['timestamp'].dt.tz) != 'UTC':
                    metric_data['timestamp'] = metric_data['timestamp'].dt.tz_convert('UTC')
                
                # 移除时区信息用于Prophet（Prophet不支持时区）
                prophet_data = pd.DataFrame({
                    'ds': metric_data['timestamp'].dt.tz_localize(None),
                    'y': metric_data['value']
                })
                
                # 检查是否有足够的数据点
                if len(prophet_data) < self.config.get('min_train_samples', 8):
                    logger.warning(f"Not enough data for metric {metric}: {len(prophet_data)} samples")
                    continue
                    
                # 创建并训练Prophet模型
                model = Prophet(
                    growth=self.config.get('growth', 'linear'),
                    daily_seasonality=self.config.get('daily_seasonality', True),
                    weekly_seasonality=self.config.get('weekly_seasonality', True),
                    yearly_seasonality=self.config.get('yearly_seasonality', False),
                    interval_width=self.config.get('interval_width', 0.95)
                )
                
                model.fit(prophet_data)
                
                # 存储训练好的模型
                self.models[metric] = model
                logger.info(f"Successfully trained model for metric {metric}")
                
            except Exception as e:
                logger.error(f"Error training model for metric {metric}: {str(e)}")
                continue
                
        # 保存训练好的模型
        if self.models:
            self.save_model()
            logger.info("All models trained and saved successfully")
    
    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """
        Make predictions for the specified horizon
        
        Args:
            horizon: Number of minutes to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.models:
            raise ValueError("No trained models available. Please train the models first.")
            
        # Convert horizon to periods (assuming 5-minute intervals)
        periods = int(horizon / 5)
        if periods < 1:
            periods = 1
            
        logger.info(f"Making predictions for {horizon} minutes ({periods} periods)")
        
        all_predictions = []
        
        # 使用UTC时间作为起始点
        start_time = datetime.now(pytz.UTC)
        
        for metric_name, model in self.models.items():
            try:
                # 创建未来时间点（不带时区）
                future_dates = pd.date_range(
                    start=start_time.replace(tzinfo=None),
                    periods=periods,
                    freq='5min'
                )
                
                # 创建未来数据框
                future = pd.DataFrame({'ds': future_dates})
                
                # 进行预测
                forecast = model.predict(future)
                
                # 获取预测的最后一个periods部分
                forecast = forecast.iloc[-periods:].copy()
                
                # 根据指标类型设置最小值和处理方式
                if metric_name in ['latency_avg', 'latency_p95', 'latency_p99']:
                    # 对于延迟指标，使用指数变换来确保正值
                    forecast['yhat'] = np.exp(forecast['yhat'])
                    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
                    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
                    
                    # 设置合理的最小值（毫秒级别）
                    min_values = {
                        'latency_avg': 0.005,    # 平均延迟最小5ms
                        'latency_p95': 0.010,    # P95延迟最小10ms
                        'latency_p99': 0.015     # P99延迟最小15ms
                    }
                    min_value = min_values.get(metric_name, 0.005)
                    
                else:
                    # 对于其他指标使用原来的最小值
                    min_values = {
                        'cpu_usage_real': 0.1,      # CPU使用率最小0.1%
                        'memory_usage_real': 1.0,    # 内存使用率最小1%
                        'network_io_real': 1.0,      # 网络I/O最小1
                        'requests_total': 0.0        # 请求数最小0
                    }
                    min_value = min_values.get(metric_name, 0)
                
                # 应用最小值限制
                forecast['yhat'] = forecast['yhat'].clip(lower=min_value)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_value)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=min_value)
                
                # 确保上界大于下界和预测值
                forecast['yhat_upper'] = forecast[['yhat_upper', 'yhat', 'yhat_lower']].max(axis=1)
                forecast['yhat_lower'] = forecast[['yhat_lower', 'yhat']].min(axis=1)
                
                # 将预测添加到结果中，添加UTC时区
                result = pd.DataFrame({
                    'timestamp': pd.to_datetime(forecast['ds']).dt.tz_localize('UTC'),
                    f'{metric_name}': forecast['yhat'],
                    f'{metric_name}_lower': forecast['yhat_lower'],
                    f'{metric_name}_upper': forecast['yhat_upper']
                })
                
                all_predictions.append(result)
                
            except Exception as e:
                logger.error(f"Error making predictions for metric {metric_name}: {str(e)}")
                
        if not all_predictions:
            raise ValueError("Failed to make predictions for any metric")
            
        # 合并所有预测
        predictions = all_predictions[0]
        for pred in all_predictions[1:]:
            predictions = predictions.merge(pred.drop('timestamp', axis=1), 
                                         left_index=True, 
                                         right_index=True)
            
        # 确保没有缺失值
        predictions = predictions.fillna(0)
        
        # 特殊处理：确保真实资源指标包含在最终结果中
        for metric in self.models.keys():
            if metric in ['cpu_usage_real', 'memory_usage_real', 'network_io_real']:
                logger.info(f"Including real system metric {metric} in predictions")
        
        logger.info(f"Final predictions columns: {predictions.columns.tolist()}")
            
        return predictions
    
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
            
    def save_model(self) -> bool:
        """
        Save trained models to disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Save each model
            for metric_name, model in self.models.items():
                model_path = models_dir / f"{metric_name}.joblib"
                joblib.dump(model, model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """
        Load trained models from disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if models directory exists
            models_dir = Path("models")
            if not models_dir.exists():
                return False
            
            # Load each model
            for model_file in models_dir.glob("*.joblib"):
                metric_name = model_file.stem
                self.models[metric_name] = joblib.load(model_file)
            
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False 