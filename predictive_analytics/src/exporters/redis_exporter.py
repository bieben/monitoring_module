"""
Redis exporter implementation
"""

import redis
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
from .base_exporter import BaseExporter
from ..config import CACHE_CONFIG
import numpy as np

logger = logging.getLogger(__name__)

class RedisExporter(BaseExporter):
    """Redis exporter for caching results"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Redis exporter
        
        Args:
            config: Redis configuration dictionary containing:
                - host: Redis host
                - port: Redis port
                - db: Redis database number
                - key_prefix: Key prefix for stored data
                - expire: TTL for stored data
        """
        if not config:
            raise ValueError("Missing required configuration")
            
        self._validate_config(config)
        super().__init__(config)
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6379),
                db=self.config.get('db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise RuntimeError(f"Redis connection failed: {str(e)}")
    
    def export(self, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Export data to Redis
        
        Args:
            data: Data to export
            key: Optional key override
            
        Returns:
            True if export successful
            
        Raises:
            ValueError: If data is invalid
            RuntimeError: If export fails
        """
        try:
            # 验证数据
            if not data:
                raise ValueError("Empty data")
            
            # 验证时间戳
            if 'timestamp' not in data:
                raise ValueError("Missing timestamp")
            try:
                pd.to_datetime(data['timestamp'])
            except:
                raise ValueError("Invalid timestamp format")
            
            # 不再验证predictions格式，接受任何有效的数据结构
            
            # 格式化数据
            formatted_data = self._format_data(data)
            
            # 序列化数据
            serialized_data = self._serialize_data(formatted_data)
            
            # 生成键
            export_key = key or self._generate_key(formatted_data)
            
            # 导出数据
            self.redis_client.set(
                export_key,
                serialized_data,
                ex=self.config.get('expire', 3600)
            )
            
            # 更新最新键
            self.redis_client.set(
                f"{self.config['key_prefix']}latest",
                export_key,
                ex=self.config.get('expire', 3600)
            )
            
            logger.info(f"Data exported successfully to key: {export_key}")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error exporting to Redis: {str(e)}")
            raise RuntimeError(f"Failed to export data: {str(e)}")
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get latest data from Redis
        
        Returns:
            Latest data or None if not found
        """
        try:
            latest_key = self.redis_client.get(f"{self.config['key_prefix']}latest")
            if not latest_key:
                return None
                
            data = self.redis_client.get(latest_key)
            if not data:
                return None
                
            return self._deserialize_data(data)
            
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical data from Redis
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of historical data
        """
        try:
            # 获取所有键
            pattern = f"{self.config['key_prefix']}*"
            keys = self.redis_client.keys(pattern)
            if not keys:
                return []
                
            # 按时间戳排序
            sorted_keys = sorted(keys, reverse=True)[:limit]
            
            # 获取数据
            history = []
            for key in sorted_keys:
                data = self.redis_client.get(key)
                if data:
                    history.append(self._deserialize_data(data))
                    
            return history
            
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    def _format_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for storage
        
        Args:
            data: Data to format
            
        Returns:
            Formatted data
        """
        formatted = data.copy()
        
        # 格式化时间戳
        if 'timestamp' not in formatted:
            raise ValueError("Missing timestamp")
        
        formatted['timestamp'] = str(pd.to_datetime(data['timestamp']))
        
        # 格式化预测数据
        if 'predictions' in formatted:
            # 支持DataFrame、字典列表和其他格式
            if isinstance(formatted['predictions'], pd.DataFrame):
                predictions_dict = formatted['predictions'].to_dict(orient='records')
                formatted['predictions'] = predictions_dict
            elif isinstance(formatted['predictions'], list):
                # 列表已经是正确格式，只需确保时间戳格式统一
                pass
            else:
                # 转换为字符串，至少确保可以序列化
                try:
                    formatted['predictions'] = str(formatted['predictions'])
                except:
                    formatted['predictions'] = "Unparseable predictions data"
                
            # 如果是列表格式，确保时间戳格式统一
            if isinstance(formatted['predictions'], list):
                for pred in formatted['predictions']:
                    if isinstance(pred, dict) and 'timestamp' in pred:
                        try:
                            pred['timestamp'] = str(pd.to_datetime(pred['timestamp']))
                        except:
                            # 如果无法解析时间戳，保持原样
                            pass
        
        # 格式化指标数据
        if 'metrics' in formatted and isinstance(formatted['metrics'], dict):
            formatted['metrics'] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in formatted['metrics'].items()
            }
        
        # 格式化优化结果
        if 'optimization' in formatted and isinstance(formatted['optimization'], dict):
            optimization = formatted['optimization']
            for k, v in optimization.items():
                if k == 'utilization' and isinstance(v, dict):
                    # 处理利用率子字典
                    for uk, uv in list(v.items()):
                        if uv is None:
                            v[uk] = 0.0
                            
        return formatted
    
    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """
        Serialize data for Redis storage
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return str(obj)
                return super().default(obj)
        
        return json.dumps(data, cls=NumpyEncoder)
    
    def _deserialize_data(self, data: str) -> Dict[str, Any]:
        """
        Deserialize data from Redis storage
        
        Args:
            data: Data to deserialize
            
        Returns:
            Deserialized data
        """
        deserialized = json.loads(data)
        
        # 转换时间戳
        if 'timestamp' in deserialized:
            deserialized['timestamp'] = str(pd.to_datetime(deserialized['timestamp']))
        
        # 转换预测数据
        if 'predictions' in deserialized and isinstance(deserialized['predictions'], list):
            for record in deserialized['predictions']:
                if 'timestamp' in record:
                    record['timestamp'] = str(pd.to_datetime(record['timestamp']))
        
        return deserialized
    
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """
        Generate Redis key for data
        
        Args:
            data: Data to generate key for
            
        Returns:
            Generated key
        """
        timestamp = pd.to_datetime(data['timestamp']).strftime('%Y%m%d%H%M%S')
        return f"{self.config['key_prefix']}{timestamp}"
    
    def _validate_config(self, config: Dict) -> bool:
        """
        Validate configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {'host', 'port', 'db', 'key_prefix'}
        if not config or not all(field in config for field in required_fields):
            raise ValueError("Missing required configuration fields")
        return True 