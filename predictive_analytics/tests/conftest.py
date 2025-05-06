"""
Pytest configuration file
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now(),
        freq='5min'
    )
    
    df = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': [random.uniform(0, 100) for _ in range(len(dates))],
        'memory_usage': [random.uniform(0, 100) for _ in range(len(dates))],
        'network_io': [random.uniform(0, 1000) for _ in range(len(dates))],
        'latency': [random.uniform(0, 500) for _ in range(len(dates))]
    })
    # 确保时间戳格式统一
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = {
        'data_validation': {
            'timestamp_format': '%Y-%m-%d %H:%M:%S',
            'value_range': {
                'cpu_usage': [0, 100],
                'memory_usage': [0, 100],
                'network_io': [0, 1000],
                'latency': [0, 1.0]
            }
        },
        'models': {
            'prophet': {
                'model_params': {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10,
                    'holidays_prior_scale': 10,
                    'seasonality_mode': 'multiplicative'
                },
                'validation': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'scoring': 'neg_mean_squared_error'
                },
                'features': ['cpu_usage', 'memory_usage', 'network_io', 'latency']
            },
            'sklearn': {
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                'validation': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'scoring': 'neg_mean_squared_error'
                },
                'features': ['hour', 'day_of_week', 'month', 'cpu_usage', 'memory_usage', 'network_io', 'latency']
            }
        },
        'optimization': {
            'constraints': {
                'max_cpu': 100,
                'max_memory': 100,
                'max_network': 1000,
                'max_latency': 1.0
            },
            'weights': {
                'cpu': 0.4,
                'memory': 0.3,
                'network': 0.3
            },
            'objective': 'minimize_cost',
            'solver_config': {
                'solver': 'CBC',
                'balance_resources': True,
                'balance_tolerance': 0.2,
                'safety_margin': 1.2,
                'min_utilization': 20,
                'max_utilization': 90
            }
        },
        'prometheus': {
            'host': 'localhost',
            'port': 9090,
            'url': 'http://localhost:9090/api/v1/query_range',
            'metrics': {
                'cpu_usage': 'container_cpu_usage_seconds_total',
                'memory_usage': 'container_memory_usage_bytes',
                'network_io': 'container_network_transmit_bytes_total',
                'latency': 'request_latency_seconds'
            },
            'retry_config': {
                'max_retries': 3,
                'retry_delay': 1,
                'timeout': 5
            },
            'max_retries': 3,
            'retry_delay': 1,
            'timeout': 5,
            'query_timeout': 30
        },
        'kafka': {
            'bootstrap_servers': ['kafka:9092'],
            'group_id': 'monitoring_group',
            'consumer_config': {
                'bootstrap_servers': ['kafka:9092'],
                'group_id': 'monitoring_group',
                'auto_offset_reset': 'earliest',
                'enable_auto_commit': True,
                'max_poll_interval_ms': 300000
            },
            'topics': {
                'metrics': {
                    'name': 'metrics',
                    'partitions': 1
                },
                'events': {
                    'name': 'events',
                    'partitions': 1
                }
            },
            'retry_config': {
                'max_retries': 3,
                'retry_delay': 1
            },
            'schema': {
                'timestamp': 'string',
                'metric_name': 'string',
                'value': 'float',
                'labels': 'dict'
            }
        },
        'cache': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'key_prefix': 'prediction:',
            'expire': 3600
        }
    }
    
    # 确保测试配置包含data_validation字段
    if 'data_validation' not in config['models']['prophet']:
        config['models']['prophet']['data_validation'] = {
            'timestamp_format': '%Y-%m-%d %H:%M:%S',
            'value_range': {
                'cpu_usage': (0, 100),
                'memory_usage': (0, 32 * 1024 * 1024 * 1024),
                'network_io': (0, 10 * 1024 * 1024 * 1024),
                'latency': (0, 1.0)
            }
        }
    
    if 'data_validation' not in config['models']['sklearn']:
        config['models']['sklearn']['data_validation'] = {
            'timestamp_format': '%Y-%m-%d %H:%M:%S',
            'value_range': {
                'cpu_usage': (0, 100),
                'memory_usage': (0, 32 * 1024 * 1024 * 1024),
                'network_io': (0, 10 * 1024 * 1024 * 1024),
                'latency': (0, 1.0)
            }
        }
        
    if 'model_params' not in config['models']['sklearn']:
        config['models']['sklearn']['model_params'] = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2
        }
    
    return config

@pytest.fixture
def sample_training_data(mock_config):
    """Create sample training data"""
    # 生成一周的数据，每5分钟一个点
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='5min'
    )
    
    df = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(0, 100, size=len(dates)),
        'memory_usage': np.random.uniform(0, 100, size=len(dates)),
        'network_io': np.random.uniform(0, 1000, size=len(dates)),
        'latency': np.random.uniform(0, 1.0, size=len(dates))  # 修改延迟范围为0-1秒
    })
    
    # 统一时间戳格式
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime(
        mock_config['data_validation']['timestamp_format']
    )
    
    return df

@pytest.fixture
def sample_data(mock_config):
    """Create sample data for testing"""
    dates = pd.date_range(
        start=datetime.now(),
        periods=5,
        freq='5min'
    )
    
    predictions = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(0, 100, size=len(dates)),
        'memory_usage': np.random.uniform(0, 100, size=len(dates)),
        'network_io': np.random.uniform(0, 1000, size=len(dates)),
        'latency': np.random.uniform(0, 1.0, size=len(dates))  # 修改延迟范围为0-1秒
    })
    
    return {
        'timestamp': datetime.now().strftime(mock_config['data_validation']['timestamp_format']),
        'predictions': predictions,
        'metrics': {
            'cpu_usage': 50.0,
            'memory_usage': 1024 * 1024 * 1024,  # 1GB
            'network_io': 100 * 1024 * 1024,  # 100MB
            'latency': 0.5  # 500ms
        }
    } 