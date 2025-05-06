"""
Factory class for creating metric collectors
"""

from typing import Dict, Optional
from .prometheus_collector import PrometheusCollector
from .kafka_collector import KafkaCollector
from ..config import PROMETHEUS_CONFIG, KAFKA_CONFIG

class CollectorFactory:
    """Factory class for creating metric collectors"""
    
    @staticmethod
    def create_collector(collector_type: str, config: Optional[Dict] = None) -> object:
        """
        Create a collector instance based on type
        
        Args:
            collector_type: Type of collector ('prometheus' or 'kafka')
            config: Optional configuration dictionary
            
        Returns:
            Collector instance
            
        Raises:
            ValueError: If collector_type is not supported or config is invalid
        """
        if not config:
            config = CollectorFactory.get_default_config(collector_type)
            
        if collector_type.lower() == 'prometheus':
            # 构建Prometheus URL
            if 'host' in config and 'port' in config:
                config['url'] = f"http://{config['host']}:{config['port']}"
            elif 'url' not in config:
                raise ValueError("Prometheus configuration must contain either 'url' or both 'host' and 'port'")
                
            return PrometheusCollector(config)
            
        elif collector_type.lower() == 'kafka':
            return KafkaCollector(config)
        else:
            raise ValueError(f"Invalid collector type: {collector_type}")
    
    @staticmethod
    def get_default_config(collector_type: str) -> Dict:
        """
        Get default configuration for a collector type
        
        Args:
            collector_type: Type of collector ('prometheus' or 'kafka')
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If collector_type is not supported
        """
        if collector_type.lower() == 'prometheus':
            return PROMETHEUS_CONFIG
        elif collector_type.lower() == 'kafka':
            return KAFKA_CONFIG
        else:
            raise ValueError(f"Invalid collector type: {collector_type}") 