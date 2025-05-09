"""
Prometheus metrics collector implementation
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from ..config import PROMETHEUS_CONFIG
import time
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class PrometheusCollector(BaseCollector):
    """Collector for Prometheus metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Prometheus collector
        
        Args:
            config: Prometheus configuration dictionary
        """
        super().__init__(config)
        
        # 验证配置
        self._validate_config()
        
        # 设置指标查询
        self.metrics = {
            'requests_total': self.config['metrics']['requests_total'],
            'latency_avg': self.config['metrics']['latency_avg'],
            'latency_p95': self.config['metrics']['latency_p95'],
            'latency_p99': self.config['metrics']['latency_p99'],
            # 添加真实资源指标
            'cpu_usage_real': self.config['metrics']['cpu_usage_real'],
            'memory_usage_real': self.config['metrics']['memory_usage_real'],
            'network_io_real': self.config['metrics']['network_io_real']
        }
        
        # 设置URL
        self.prometheus_url = self.config['url']
        
        # 设置重试参数
        self.retry_config = {
            'max_retries': self.config.get('retry_attempts', 3),
            'retry_delay': self.config.get('retry_delay', 1),
            'timeout': self.config.get('query_timeout', 30)
        }
    
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {'url', 'metrics'}
        if not all(field in self.config for field in required_fields):
            raise ValueError("Missing required configuration fields")
            
        required_metrics = {'requests_total', 'latency_avg', 'latency_p95', 'latency_p99', 
                           'cpu_usage_real', 'memory_usage_real', 'network_io_real'}
        if not all(metric in self.config['metrics'] for metric in required_metrics):
            logger.warning("Some real system metrics are missing in configuration. Only application metrics will be collected.")
            # 确保至少有基本的应用指标
            basic_metrics = {'requests_total', 'latency_avg', 'latency_p95', 'latency_p99'}
            if not all(metric in self.config['metrics'] for metric in basic_metrics):
                raise ValueError("Missing required basic metrics in configuration")
            
        return True
    
    def collect_metrics(self, start_time: datetime, end_time: datetime, step: str = '15s') -> pd.DataFrame:
        """
        Collect metrics from Prometheus
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            step: Time resolution for data points
            
        Returns:
            DataFrame containing collected metrics
            
        Raises:
            ValueError: If time range is invalid
            RuntimeError: If all retry attempts fail
        """
        if end_time <= start_time:
            raise ValueError("End time must be after start time")
            
        all_metrics = []
        errors = []
        
        # 跟踪找到的真实资源指标
        real_metrics_found = []
        
        for metric_name, query in self.metrics.items():
            try:
                metric_data = self._query_prometheus_with_retry(query, start_time, end_time, step)
                if metric_data is not None:
                    metric_df = self._process_metric_data(metric_data, metric_name)
                    all_metrics.append(metric_df)
                    
                    # 跟踪真实资源指标
                    if metric_name in ['cpu_usage_real', 'memory_usage_real', 'network_io_real']:
                        real_metrics_found.append(metric_name)
                        logger.info(f"Successfully collected real system metric: {metric_name}")
                else:
                    errors.append(f"Failed to collect metric {metric_name}: No data available")
            except Exception as e:
                errors.append(f"Failed to collect metric {metric_name}: {str(e)}")
                logger.error(f"Error collecting metric {metric_name}: {str(e)}")
        
        if not all_metrics:
            error_msg = "; ".join(errors)
            raise RuntimeError(f"Failed to collect any metrics: {error_msg}")
        
        # 创建一个包含所有数据的单一DataFrame
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # 确保没有重复列
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # 验证数据
        if not self._validate_data(combined_df):
            raise RuntimeError("Invalid metric data format")
        
        logger.info(f"Collected metrics with shape {combined_df.shape}, including real system metrics: {real_metrics_found}")
        logger.info(f"Unique metric names in collected data: {combined_df['metric_name'].unique()}")
        
        return combined_df
    
    def _query_prometheus_with_retry(self, query: str, start_time: datetime, end_time: datetime, step: str) -> Optional[Dict]:
        """
        Query Prometheus API with retry mechanism
        
        Args:
            query: PromQL query string
            start_time: Start time
            end_time: End time
            step: Time resolution
            
        Returns:
            Dictionary containing the Prometheus response
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        attempt = 0
        last_error = None
        
        while attempt < self.retry_config['max_retries']:
            try:
                return self._query_prometheus(query, start_time, end_time, step)
            except requests.exceptions.RequestException as e:
                last_error = e
                attempt += 1
                if attempt < self.retry_config['max_retries']:
                    wait_time = self.retry_config['retry_delay'] * (2 ** attempt)
                    logger.warning(f"Retry attempt {attempt} after {wait_time} seconds")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Failed after {self.retry_config['max_retries']} attempts: {str(last_error)}")
    
    def _query_prometheus(self, query: str, start_time: datetime, end_time: datetime, step: str) -> Optional[Dict]:
        """
        Query Prometheus API
        
        Args:
            query: PromQL query string
            start_time: Start time
            end_time: End time
            step: Time resolution
            
        Returns:
            Dictionary containing the Prometheus response
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response format is invalid
        """
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
        
        try:
            response = requests.get(
                self.prometheus_url,
                params=params,
                timeout=self.retry_config['timeout']
            )
            response.raise_for_status()
            result = response.json()
            
            if result['status'] != 'success':
                raise ValueError(f"Query failed: {result.get('error', 'Unknown error')}")
                
            if not result['data']['result']:
                # 对于真实系统指标，记录警告但不将其视为错误
                if query in [self.config['metrics'].get('cpu_usage_real'), 
                            self.config['metrics'].get('memory_usage_real'), 
                            self.config['metrics'].get('network_io_real')]:
                    logger.warning(f"No real system metric data returned for query: {query}. This may be expected if Node Exporter is not properly configured.")
                else:
                    logger.warning(f"No data returned for query: {query}")
                return None
                
            return result['data']['result']
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Prometheus: {str(e)}")
            raise
    
    def _process_metric_data(self, metric_data: List[Dict], metric_name: str) -> pd.DataFrame:
        """
        Process raw metric data into a DataFrame
        
        Args:
            metric_data: Raw metric data from Prometheus
            metric_name: Name of the metric
            
        Returns:
            DataFrame with processed metric data
        """
        processed_data = []
        
        for result in metric_data:
            # 提取标签信息
            labels = result.get('metric', {})
            
            # 根据不同类型的指标提取相关标签
            if metric_name in ['cpu_usage_real', 'memory_usage_real', 'network_io_real']:
                # 对于Node Exporter指标，使用instance作为标识
                model_id = 'system'  # 系统级别指标
                instance = labels.get('instance', 'unknown')
                # 可以在这里添加额外的标签处理逻辑
                extra_labels = {'instance': instance}
                
                # 特别记录这些是真实系统指标
                logger.info(f"Processing real system metric: {metric_name} from instance {instance}")
            else:
                # 对于应用指标，使用model_id作为标识
                model_id = labels.get('model_id', 'unknown')
                extra_labels = {}
            
            # 处理数据点
            for timestamp, value in result['values']:
                try:
                    data_point = {
                        'timestamp': pd.to_datetime(timestamp, unit='s'),
                        'metric_name': metric_name,
                        'value': float(value),
                        'model_id': model_id,
                    }
                    # 添加额外标签
                    data_point.update(extra_labels)
                    processed_data.append(data_point)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing data point: {str(e)}")
                    continue
        
        if not processed_data:
            # 如果没有数据，返回空DataFrame
            return pd.DataFrame(columns=['timestamp', 'metric_name', 'value', 'model_id'])
            
        df = pd.DataFrame(processed_data)
        
        # 统一时间戳格式
        if 'data_validation' in self.config and 'timestamp_format' in self.config['data_validation']:
            df['timestamp'] = df['timestamp'].dt.strftime(self.config['data_validation']['timestamp_format'])
        
        # 处理重复标签 - 为每个时间戳、指标、模型组合添加一个唯一索引
        if len(df) > 0:
            # 确保没有重复的索引
            df = df.reset_index(drop=True)
            
            # 如果有重复行，则按时间戳、指标名称和模型ID分组，取平均值
            group_cols = ['timestamp', 'metric_name', 'model_id']
            if 'instance' in df.columns:
                group_cols.append('instance')
                
            if df.duplicated(group_cols).any():
                df = df.groupby(group_cols).agg({
                    'value': 'mean'
                }).reset_index()
        
        return df
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics from Prometheus
        
        Returns:
            List of metric names
        """
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/label/__name__/values",
                timeout=self.retry_config['timeout']
            )
            response.raise_for_status()
            result = response.json()
            
            if result['status'] == 'success':
                return result['data']
            else:
                logger.warning("Failed to get available metrics")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting available metrics: {str(e)}")
            return []
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据格式是否正确
        
        Args:
            data: 要验证的数据
            
        Returns:
            True 如果数据格式正确，否则 False
        """
        if data is None or data.empty:
            logger.warning("Empty dataframe during data validation")
            return False
            
        # 检查必要的列
        required_columns = {'timestamp', 'metric_name', 'value', 'model_id'}
        if not all(col in data.columns for col in required_columns):
            missing = required_columns - set(data.columns)
            logger.warning(f"Missing required columns: {missing}")
            return False
            
        # 验证值的范围
        if 'data_validation' in self.config and 'value_range' in self.config['data_validation']:
            for metric_name, (min_val, max_val) in self.config['data_validation']['value_range'].items():
                metric_data = data[data['metric_name'] == metric_name]
                if not metric_data.empty:
                    # 检查是否有超出范围的值
                    out_of_range = metric_data[(metric_data['value'] < min_val) | (metric_data['value'] > max_val)]
                    if not out_of_range.empty:
                        logger.warning(f"Found {len(out_of_range)} values out of range for metric {metric_name}")
                        # 修剪超出范围的值
                        data.loc[data['metric_name'] == metric_name, 'value'] = data.loc[data['metric_name'] == metric_name, 'value'].clip(min_val, max_val)
        
        return True 