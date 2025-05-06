"""
Tests for resource optimizers
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.optimizers import PuLPOptimizer, OptimizerFactory

@pytest.fixture
def sample_predictions():
    """Create sample prediction data"""
    dates = pd.date_range(
        start=datetime.now(),
        periods=30,
        freq='5min'
    )
    
    return pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(40, 80, size=len(dates)),
        'memory_usage': np.random.uniform(50, 90, size=len(dates)),
        'network_io': np.random.uniform(200, 800, size=len(dates)),
        'latency': np.random.uniform(100, 400, size=len(dates))
    })

def test_pulp_optimizer(sample_predictions, mock_config):
    """Test PuLP optimizer"""
    optimizer = PuLPOptimizer(mock_config['optimization'])
    
    result = optimizer.optimize(
        predictions=sample_predictions,
        constraints=mock_config['optimization']['constraints']
    )
    
    assert isinstance(result, dict)
    assert 'cpu_allocation' in result
    assert 'memory_allocation' in result
    assert 'network_allocation' in result
    assert 'status' in result
    assert 'utilization' in result
    
    # 确保分配的资源是浮点数
    assert isinstance(result['cpu_allocation'], (float, int, np.number))
    assert isinstance(result['memory_allocation'], (float, int, np.number))
    assert isinstance(result['network_allocation'], (float, int, np.number))
    
    # 检查约束
    constraints = mock_config['optimization']['constraints']
    assert result['cpu_allocation'] <= constraints['max_cpu']
    assert result['memory_allocation'] <= constraints['max_memory']
    assert result['network_allocation'] <= constraints['max_network']

def test_optimizer_factory(mock_config):
    """Test optimizer factory"""
    # Test PuLP optimizer creation
    pulp_optimizer = OptimizerFactory.create_optimizer('pulp', mock_config['optimization'])
    assert isinstance(pulp_optimizer, PuLPOptimizer)
    
    # Test invalid optimizer type
    with pytest.raises(ValueError):
        OptimizerFactory.create_optimizer('invalid', {})

def test_optimization_with_invalid_predictions(mock_config):
    """Test optimization with invalid prediction data"""
    optimizer = PuLPOptimizer(mock_config['optimization'])
    constraints = mock_config['optimization']['constraints']
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        optimizer.optimize(predictions=pd.DataFrame(), constraints=constraints)
    
    # Test with missing required columns
    invalid_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=5, freq='5min'),
        'some_metric': np.random.uniform(0, 100, size=5)
    })
    with pytest.raises(ValueError):
        optimizer.optimize(predictions=invalid_data, constraints=constraints)

def test_optimization_with_extreme_values(mock_config):
    """Test optimization with extreme prediction values"""
    optimizer = PuLPOptimizer(mock_config['optimization'])
    constraints = mock_config['optimization']['constraints']
    
    # Create predictions with extreme values
    dates = pd.date_range(start=datetime.now(), periods=5, freq='5min')
    extreme_predictions = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': [150, 160, 170, 180, 190],  # Values above max constraint
        'memory_usage': [95, 96, 97, 98, 99],
        'network_io': [1200, 1300, 1400, 1500, 1600],
        'latency': [500, 550, 600, 650, 700]
    })
    
    result = optimizer.optimize(predictions=extreme_predictions, constraints=constraints)
    
    # Check that optimal values respect constraints
    assert result['cpu_allocation'] <= constraints['max_cpu']
    assert result['memory_allocation'] <= constraints['max_memory']
    assert result['network_allocation'] <= constraints['max_network']

def test_optimization_with_different_objectives(mock_config, sample_predictions):
    """Test optimization with different objective functions"""
    base_config = mock_config['optimization'].copy()
    constraints = base_config['constraints']
    
    # Test with different optimization objectives
    objectives = ['minimize_cost', 'maximize_performance', 'balance']
    for objective in objectives:
        config = base_config.copy()
        config['objective'] = objective
        optimizer = PuLPOptimizer(config)
        
        result = optimizer.optimize(predictions=sample_predictions, constraints=constraints)
        assert isinstance(result, dict)
        assert 'cpu_allocation' in result
        assert 'memory_allocation' in result
        assert 'network_allocation' in result
        assert 'status' in result
        assert 'utilization' in result
        assert 'objective_value' in result 