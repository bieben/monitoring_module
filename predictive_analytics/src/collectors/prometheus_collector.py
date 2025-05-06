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
            'cpu_usage': self.config['metrics']['cpu_usage'],
            'memory_usage': self.config['metrics']['memory_usage'],
            'network_io': self.config['metrics']['network_io'],
            'latency': self.config['metrics']['latency']
        }
        
        # 设置URL
        self.prometheus_url = self.config['url']
        
        # 设置重试参数
        self.retry_config = {
            'max_retries': self.config.get('max_retries', 3),
            'retry_delay': self.config.get('retry_delay', 1),
            'timeout': self.config.get('timeout', 10)
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
            
        required_metrics = {'cpu_usage', 'memory_usage', 'network_io', 'latency'}
        if not all(metric in self.config['metrics'] for metric in required_metrics):
            raise ValueError("Missing required metrics in configuration")
            
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
        
        for metric_name, query in self.metrics.items():
            try:
                metric_data = self._query_prometheus_with_retry(query, start_time, end_time, step)
                if metric_data is not None:
                    metric_df = self._process_metric_data(metric_data, metric_name)
                    all_metrics.append(metric_df)
                else:
                    errors.append(f"Failed to collect metric {metric_name}: No data available")
            except Exception as e:
                errors.append(f"Failed to collect metric {metric_name}: {str(e)}")
                logger.error(f"Error collecting metric {metric_name}: {str(e)}")
        
        if not all_metrics:
            error_msg = "; ".join(errors)
            raise RuntimeError(f"Failed to collect any metrics: {error_msg}")
        
        # 合并所有指标数据
        result = pd.concat(all_metrics, axis=1)
        
        # 验证数据
        if not self._validate_data(result):
            raise RuntimeError("Invalid metric data format")
        
        return result
    
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
            
            # 处理数据点
            for timestamp, value in result['values']:
                try:
                    processed_data.append({
                        'timestamp': pd.to_datetime(timestamp, unit='s'),
                        'metric_name': metric_name,
                        'value': float(value),
                        'labels': labels
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing data point: {str(e)}")
                    continue
        
        df = pd.DataFrame(processed_data)
        
        # 统一时间戳格式
        if 'data_validation' in self.config and 'timestamp_format' in self.config['data_validation']:
            df['timestamp'] = df['timestamp'].dt.strftime(self.config['data_validation']['timestamp_format'])
        
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