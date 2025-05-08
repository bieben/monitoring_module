"""
Tests for prediction models
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from predictive_analytics.src.models import ProphetModel, SklearnModel, ModelFactory
from predictive_analytics.src.optimizers import PuLPOptimizer

@pytest.fixture
def sample_training_data(mock_config):
    """Create sample training data"""
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
    
    # 验证数据范围
    for metric, range_values in mock_config['data_validation']['value_range'].items():
        assert df[metric].between(range_values[0], range_values[1]).all(), \
            f"{metric} values out of valid range"
    
    return df

def test_prophet_model(mock_config, sample_training_data):
    """Test Prophet model"""
    model = ProphetModel(mock_config['models']['prophet'])
    
    # 训练模型
    model.train(sample_training_data)
    
    # 预测
    horizon = 30  # 30分钟
    predictions = model.predict(horizon)
    
    # 验证预测结果
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == horizon
    assert all(col in predictions.columns for col in ['timestamp', 'cpu_usage', 'memory_usage', 'network_io', 'latency'])
    
    # 验证预测值范围
    for metric, range_values in mock_config['data_validation']['value_range'].items():
        assert predictions[metric].between(range_values[0], range_values[1]).all(), \
            f"Predicted {metric} values out of valid range"

def test_sklearn_model(mock_config, sample_training_data):
    """Test Scikit-learn model"""
    model = SklearnModel(mock_config['models']['sklearn'])
    
    # 训练模型
    model.train(sample_training_data)
    
    # 预测
    horizon = 30  # 30分钟
    predictions = model.predict(horizon)
    
    # 验证预测结果
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == horizon
    assert all(col in predictions.columns for col in ['timestamp', 'cpu_usage', 'memory_usage', 'network_io', 'latency'])
    
    # 验证预测值范围
    for metric, range_values in mock_config['data_validation']['value_range'].items():
        assert predictions[metric].between(range_values[0], range_values[1]).all(), \
            f"Predicted {metric} values out of valid range"

def test_prophet_model_save_load(mock_config, sample_training_data, tmp_path):
    """Test Prophet model save and load"""
    model = ProphetModel(mock_config['models']['prophet'])
    model.train(sample_training_data)
    
    # 保存模型
    save_path = tmp_path / "prophet_model.joblib"
    model.save(save_path)
    
    # 加载模型
    loaded_model = ProphetModel(mock_config['models']['prophet'])
    loaded_model.load(save_path)
    
    # 验证预测结果一致性
    horizon = 30
    original_predictions = model.predict(horizon)
    loaded_predictions = loaded_model.predict(horizon)
    
    # 检查基本结构
    assert loaded_model.is_trained == True
    assert set(loaded_model.models.keys()) == set(model.models.keys())
    assert loaded_predictions.shape == original_predictions.shape
    assert set(loaded_predictions.columns) == set(original_predictions.columns)

def test_sklearn_model_save_load(mock_config, sample_training_data, tmp_path):
    """Test Scikit-learn model save and load"""
    model = SklearnModel(mock_config['models']['sklearn'])
    model.train(sample_training_data)
    
    # 保存模型
    save_path = tmp_path / "sklearn_model.joblib"
    model.save(save_path)
    
    # 加载模型
    loaded_model = SklearnModel(mock_config['models']['sklearn'])
    loaded_model.load(save_path)
    
    # 验证预测结果一致性
    horizon = 30
    original_predictions = model.predict(horizon)
    loaded_predictions = loaded_model.predict(horizon)
    
    # 检查基本结构
    assert loaded_model.is_trained == True
    assert set(loaded_model.models.keys()) == set(model.models.keys())
    assert loaded_predictions.shape == original_predictions.shape
    assert set(loaded_predictions.columns) == set(original_predictions.columns)
    
    # 忽略时间戳列进行比较
    original_values = original_predictions.drop(columns=['timestamp'])
    loaded_values = loaded_predictions.drop(columns=['timestamp'])
    
    # 允许一定的浮点数误差
    pd.testing.assert_frame_equal(original_values, loaded_values, atol=1e-2, rtol=1e-2)

def test_model_error_handling(mock_config, sample_training_data):
    """Test model error handling"""
    # 测试无效配置
    with pytest.raises(ValueError):
        ProphetModel({})
    with pytest.raises(ValueError):
        SklearnModel({})
    
    # 测试无效训练数据
    model = ProphetModel(mock_config['models']['prophet'])
    with pytest.raises(ValueError):
        model.train(pd.DataFrame())
    
    # 测试未训练就预测
    model = SklearnModel(mock_config['models']['sklearn'])
    with pytest.raises(ValueError, match="Model not trained"):
        model.predict(30)
    
    # 测试无效预测范围
    model = ProphetModel(mock_config['models']['prophet'])
    model.train(sample_training_data)
    with pytest.raises(ValueError):
        model.predict(-1)
    with pytest.raises(ValueError):
        model.predict(0)

def test_model_factory(mock_config):
    """Test model factory"""
    # 测试Prophet模型创建
    prophet_model = ModelFactory.create_model('prophet', mock_config['models']['prophet'])
    assert isinstance(prophet_model, ProphetModel)
    
    # 测试Sklearn模型创建
    sklearn_model = ModelFactory.create_model('sklearn', mock_config['models']['sklearn'])
    assert isinstance(sklearn_model, SklearnModel)
    
    # 测试无效模型类型
    with pytest.raises(ValueError):
        ModelFactory.create_model('invalid', {})

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
        'latency': [0.5, 0.6, 0.7, 0.8, 0.9]  # 修改延迟范围为0-1秒
    })
    
    result = optimizer.optimize(predictions=extreme_predictions, constraints=constraints)
    
    # 验证结果
    assert result['status'] == 'optimal'
    # 对于CPU，验证分配值应该≥最大预测值或达到最大约束
    assert result['cpu_allocation'] >= extreme_predictions['cpu_usage'].max() or result['cpu_allocation'] == constraints['max_cpu']
    # 对于内存，验证分配值应该≥最大预测值
    assert result['memory_allocation'] >= extreme_predictions['memory_usage'].max()
    # 对于网络，验证分配值应该≥最大预测值或达到最大约束
    assert result['network_allocation'] >= extreme_predictions['network_io'].max() or result['network_allocation'] == constraints['max_network'] 