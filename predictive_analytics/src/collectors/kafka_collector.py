"""
Kafka metrics collector implementation
"""

import pandas as pd
import json
from kafka import KafkaConsumer
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from ..config import KAFKA_CONFIG
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class KafkaCollector(BaseCollector):
    """Collector for Kafka metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Kafka collector
        
        Args:
            config: Kafka configuration dictionary
        """
        super().__init__(config)
        
        # 验证配置
        self._validate_config()
        
        # 设置主题和配置
        self.topics = [self.config['topics']] if isinstance(self.config['topics'], str) else self.config['topics']
        
        # 设置消费者配置
        self.consumer_config = {
            'bootstrap_servers': self.config.get('bootstrap_servers', 'localhost:9092'),
            'group_id': self.config.get('group_id', 'monitoring_group'),
            'auto_offset_reset': self.config.get('auto_offset_reset', 'earliest'),
            'enable_auto_commit': self.config.get('enable_auto_commit', True),
            'max_poll_records': self.config.get('max_poll_records', 500),
            'max_poll_interval_ms': self.config.get('max_poll_interval_ms', 300000)
        }
        
        # 初始化schema
        self.schema = self.config.get('schema', {})
        
        self._consumer = None
    
    def _validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {'topics', 'bootstrap_servers', 'group_id'}
        if not all(field in self.config for field in required_fields):
            raise ValueError("Missing required configuration fields")
        return True
    
    @property
    def consumer(self):
        """
        Get or create Kafka consumer
        
        Returns:
            KafkaConsumer instance
        """
        if self._consumer is None:
            try:
                self._consumer = KafkaConsumer(
                    *self.topics,
                    **self.consumer_config
                )
            except Exception as e:
                logger.error(f"Failed to create Kafka consumer: {str(e)}")
                raise RuntimeError(f"Failed to create Kafka consumer: {str(e)}")
        return self._consumer
    
    def collect_metrics(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Collect metrics from Kafka topics
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            DataFrame containing collected metrics
            
        Raises:
            ValueError: If time range is invalid
            RuntimeError: If Kafka operations fail
        """
        if end_time <= start_time:
            raise ValueError("End time must be after start time")
            
        metrics_data = []
        errors = []
        
        try:
            # 设置每个分区的起始位置
            for topic_partition in self.consumer.assignment():
                try:
                    self.consumer.seek_to_beginning(topic_partition)
                except Exception as e:
                    errors.append(f"Failed to seek partition {topic_partition}: {str(e)}")
                    logger.error(f"Error seeking partition {topic_partition}: {str(e)}")
            
            # 收集指定时间范围内的消息
            poll_timeout = self.consumer_config.get('max_poll_interval_ms', 300000) / 1000
            messages = self.consumer.poll(timeout_ms=poll_timeout)
            
            for topic_partition, msgs in messages.items():
                for message in msgs:
                    try:
                        # 验证消息格式
                        if not self._validate_message_format(message.value):
                            logger.warning(f"Invalid message format: {message.value}")
                            continue
                        
                        timestamp = datetime.fromtimestamp(message.timestamp / 1000.0)
                        
                        # 跳过时间范围外的消息
                        if timestamp < start_time:
                            continue
                        if timestamp > end_time:
                            break
                        
                        # 处理消息
                        metric_data = self._process_message(message.value, timestamp)
                        if metric_data:
                            metrics_data.extend(metric_data)
                        
                    except Exception as e:
                        errors.append(f"Failed to process message: {str(e)}")
                        logger.error(f"Error processing message: {str(e)}")
            
        except Exception as e:
            errors.append(f"Failed to collect metrics from Kafka: {str(e)}")
            logger.error(f"Error collecting metrics from Kafka: {str(e)}")
        finally:
            self._close_consumer()
        
        if not metrics_data:
            error_msg = "; ".join(errors) if errors else "No data available"
            raise RuntimeError(f"Failed to collect metrics: {error_msg}")
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics_data)
        
        # 验证数据
        if not self._validate_data(df):
            raise RuntimeError("Invalid metric data format")
        
        return df
    
    def _validate_message_format(self, message: Dict) -> bool:
        """
        Validate message format against schema
        
        Args:
            message: Message to validate
            
        Returns:
            bool: True if message is valid, False otherwise
        """
        try:
            # 验证必需字段
            required_fields = self.schema.keys()
            if not all(field in message for field in required_fields):
                return False
            
            # 验证字段类型
            for field, field_type in self.schema.items():
                value = message[field]
                if field_type == 'string' and not isinstance(value, str):
                    return False
                elif field_type == 'float' and not isinstance(value, (int, float)):
                    return False
                elif field_type == 'dict' and not isinstance(value, dict):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Message validation failed: {str(e)}")
            return False
    
    def _process_message(self, message: Dict, timestamp: datetime) -> List[Dict]:
        """
        Process a Kafka message into metric data
        
        Args:
            message: Kafka message value
            timestamp: Message timestamp
            
        Returns:
            List of processed metric dictionaries
        """
        processed_data = []
        
        try:
            metric_name = message['metric_name']
            value = float(message['value'])
            labels = message.get('labels', {})
            
            processed_data.append({
                'timestamp': timestamp,
                'metric_name': metric_name,
                'value': value,
                'labels': labels
            })
                
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing message: {str(e)}")
            
        return processed_data
    
    def _close_consumer(self):
        """Close the Kafka consumer if it exists"""
        if self._consumer is not None:
            try:
                self._consumer.close()
                self._consumer = None
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {str(e)}")
    
    def __del__(self):
        """Ensure consumer is closed when object is destroyed"""
        self._close_consumer() 