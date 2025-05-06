"""
Tests for exporters
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.exporters import RedisExporter, ExporterFactory
import redis
import json
from typing import Optional, Dict

@pytest.fixture
def mock_redis_client(mocker):
    """Mock Redis client"""
    mock_client = mocker.Mock()
    mock_client.ping.return_value = True
    mock_client.set.return_value = True
    
    # 创建包含预测数据的mock响应
    predictions_list = []
    for i in range(5):
        predictions_list.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_usage': 50.0 + i,
            'memory_usage': (1024 + i) * 1024 * 1024,
            'network_io': (100 + i) * 1024 * 1024,
            'latency': 0.5 + i * 0.1
        })
        
    mock_response = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'predictions': predictions_list,
        'metrics': {
            'cpu_usage': 50.0,
            'memory_usage': 1024 * 1024 * 1024,
            'network_io': 100 * 1024 * 1024,
            'latency': 0.5
        }
    }
    
    mock_client.get.return_value = json.dumps(mock_response)
    mock_client.keys.return_value = ['test_key']
    return mocker.patch('redis.Redis', return_value=mock_client)

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
        'memory_usage': np.random.uniform(0, 32 * 1024 * 1024 * 1024, size=len(dates)),
        'network_io': np.random.uniform(0, 10 * 1024 * 1024 * 1024, size=len(dates)),
        'latency': np.random.uniform(0, 1.0, size=len(dates))  # 修改延迟范围为0-1秒
    })
    
    # 统一时间戳格式
    predictions['timestamp'] = predictions['timestamp'].dt.strftime(
        mock_config['data_validation']['timestamp_format']
    )
    
    data = {
        'timestamp': datetime.now().strftime(mock_config['data_validation']['timestamp_format']),
        'predictions': predictions,
        'metrics': {
            'cpu_usage': 50.0,
            'memory_usage': 1024 * 1024 * 1024,
            'network_io': 100 * 1024 * 1024,
            'latency': 0.5
        }
    }
    
    return data

def test_redis_exporter(mock_redis_client, mock_config, sample_data):
    """Test Redis exporter"""
    exporter = RedisExporter(mock_config['cache'])
    
    # 测试导出
    assert exporter.export(sample_data)
    
    # 测试获取最新数据
    latest = exporter.get_latest()
    assert latest is not None
    assert 'predictions' in latest
    assert 'metrics' in latest
    
    # 验证时间戳格式
    assert isinstance(latest['timestamp'], str)
    pd.to_datetime(latest['timestamp'])  # 确保可以解析
    
    # 验证预测数据
    predictions = latest['predictions']
    assert isinstance(predictions, list)  # JSON序列化后是列表
    assert len(predictions) > 0
    
    # 验证指标数据
    metrics = latest['metrics']
    assert isinstance(metrics, dict)
    assert all(isinstance(v, (int, float)) for v in metrics.values())

def test_redis_exporter_error_handling(mocker, mock_config):
    """Test Redis exporter error handling"""
    # 测试无效配置
    with pytest.raises(ValueError, match="Missing required configuration"):
        RedisExporter({})
    
    # 保存原始的__init__方法
    original_init = RedisExporter.__init__
    
    # 定义一个新的__init__方法来避免实际连接Redis
    def mock_init(self, config: Optional[Dict] = None):
        if not config:
            raise ValueError("Missing required configuration")
        
        self._validate_config(config)
        # 调用BaseExporter的init方法
        from src.exporters.base_exporter import BaseExporter
        BaseExporter.__init__(self, config)
        
        # 创建一个会抛出异常的mock Redis客户端
        mock_client = mocker.Mock()
        mock_client.ping.return_value = True  # 这样初始化不会失败
        mock_client.set.side_effect = redis.ConnectionError("Connection failed")
        self.redis_client = mock_client
    
    try:
        # 替换__init__方法
        RedisExporter.__init__ = mock_init
        
        # 创建导出器
        exporter = RedisExporter(mock_config['cache'])
        
        # 测试导出时会引发异常
        with pytest.raises(RuntimeError):
            exporter.export({'timestamp': datetime.now().isoformat()})
            
    finally:
        # 恢复原始方法
        RedisExporter.__init__ = original_init

def test_redis_exporter_with_invalid_data(mock_redis_client, mock_config):
    """Test Redis exporter with invalid data"""
    exporter = RedisExporter(mock_config['cache'])
    
    # 测试空数据
    with pytest.raises(ValueError, match="Empty data"):
        exporter.export({})
    
    # 测试无效时间戳
    invalid_data = {
        'timestamp': 'invalid_timestamp',
        'predictions': pd.DataFrame(),
        'metrics': {}
    }
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        exporter.export(invalid_data)
    
    # 测试无效预测数据
    invalid_data = {
        'timestamp': datetime.now().strftime(mock_config['data_validation']['timestamp_format']),
        'predictions': 'not_a_dataframe',
        'metrics': {}
    }
    with pytest.raises(ValueError, match="Invalid predictions format"):
        exporter.export(invalid_data)

def test_redis_exporter_data_serialization(mock_redis_client, mock_config, sample_data):
    """Test Redis exporter data serialization"""
    exporter = RedisExporter(mock_config['cache'])
    
    # 测试导出
    assert exporter.export(sample_data)
    
    # 获取数据并验证序列化
    latest = exporter.get_latest()
    assert latest is not None
    
    # 验证时间戳
    assert isinstance(latest['timestamp'], str)
    timestamp = pd.to_datetime(latest['timestamp'])
    assert isinstance(timestamp, pd.Timestamp)
    
    # 验证预测数据
    predictions = latest['predictions']
    assert isinstance(predictions, list)  # JSON序列化后是列表
    assert len(predictions) > 0
    for pred in predictions:
        assert isinstance(pred['timestamp'], str)
        assert all(isinstance(v, (int, float)) for k, v in pred.items() if k != 'timestamp')
    
    # 验证指标数据
    metrics = latest['metrics']
    assert isinstance(metrics, dict)
    assert all(isinstance(v, (int, float)) for v in metrics.values())

def test_exporter_factory(mock_config):
    """Test exporter factory"""
    # 测试Redis导出器创建
    redis_exporter = ExporterFactory.create_exporter('redis', mock_config['cache'])
    assert isinstance(redis_exporter, RedisExporter)
    
    # 测试无效导出器类型
    with pytest.raises(ValueError, match="Unsupported exporter type"):
        ExporterFactory.create_exporter('invalid', {}) 