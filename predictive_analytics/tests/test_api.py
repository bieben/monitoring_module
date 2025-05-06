"""
Tests for the REST API endpoints
"""

import pytest
from datetime import datetime, timedelta
from src.api import create_app
import json
import pandas as pd
import numpy as np
from flask import Flask
from src.models.prophet_model import ProphetModel

@pytest.fixture
def mock_metrics_data():
    """Create mock metrics data"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now(),
        freq='5min'
    )
    
    return pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(0, 100, size=len(dates)),
        'memory_usage': np.random.uniform(0, 100, size=len(dates)),
        'network_io': np.random.uniform(0, 1000, size=len(dates)),
        'latency': np.random.uniform(0, 500, size=len(dates))  # 添加延迟指标
    })

@pytest.fixture
def client(mocker, mock_metrics_data):
    """Create a test client with mocked dependencies"""
    # Mock Prometheus collector
    mocker.patch('src.collectors.prometheus_collector.PrometheusCollector.collect_metrics',
                 return_value=mock_metrics_data)
    
    # Mock model training
    mocker.patch('src.models.prophet_model.ProphetModel.train',
                 return_value=None)
    
    # Mock model predictions
    mock_predictions = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=30, freq='5min'),
        'cpu_usage': np.random.uniform(0, 100, size=30),
        'memory_usage': np.random.uniform(0, 100, size=30),
        'network_io': np.random.uniform(0, 1000, size=30),
        'latency': np.random.uniform(0, 500, size=30),
        'cpu_usage_lower': np.random.uniform(0, 50, size=30),
        'cpu_usage_upper': np.random.uniform(50, 100, size=30),
        'memory_usage_lower': np.random.uniform(0, 50, size=30),
        'memory_usage_upper': np.random.uniform(50, 100, size=30),
        'network_io_lower': np.random.uniform(0, 500, size=30),
        'network_io_upper': np.random.uniform(500, 1000, size=30),
        'latency_lower': np.random.uniform(0, 250, size=30),
        'latency_upper': np.random.uniform(250, 500, size=30)
    })
    
    # 将时间戳转换为字符串格式
    mock_predictions['timestamp'] = mock_predictions['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    mocker.patch('src.models.prophet_model.ProphetModel.predict',
                 return_value=mock_predictions)
    
    # Mock model save/load
    mocker.patch.object(ProphetModel, 'save', return_value=None)
    mocker.patch.object(ProphetModel, 'load', return_value=None)
    
    # Mock optimization results
    mock_optimization = {
        'status': 'optimal',
        'cpu_allocation': 80,
        'memory_allocation': 75,
        'network_allocation': 800,
        'latency': 200,
        'utilization': 0.85
    }
    mocker.patch('src.optimizers.pulp_optimizer.PuLPOptimizer.optimize',
                 return_value=mock_optimization)
    
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_get_current_metrics(client):
    """Test getting current metrics"""
    response = client.get('/api/v1/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'data' in data
    assert len(data['data']) > 0

def test_predict(client):
    """Test prediction endpoint"""
    test_data = {
        'horizon': 30,
        'use_cache': False
    }
    response = client.post('/api/v1/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'predictions' in data['data']
    assert 'optimization' in data['data']
    assert len(data['data']['predictions']) > 0

def test_get_latest_prediction(client):
    """Test getting latest prediction"""
    response = client.get('/api/v1/predictions/latest')
    assert response.status_code in [200, 404]  # 404 if no predictions available
    data = json.loads(response.data)
    if response.status_code == 200:
        assert data['status'] == 'success'
        assert 'data' in data
    else:
        assert data['status'] == 'error'
        assert 'message' in data

def test_get_prediction_history(client):
    """Test getting prediction history"""
    response = client.get('/api/v1/predictions/history?limit=5')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'data' in data 