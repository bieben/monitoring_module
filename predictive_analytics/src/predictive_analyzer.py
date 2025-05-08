"""
Predictive Analytics Module for AI Model Resource Usage Forecasting
This module provides functionality for collecting metrics, making predictions,
and optimizing resource allocation based on time-series data.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from pulp import *
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Union, Optional
import os

# Import collectors
from .collectors.base_collector import BaseCollector
from .collectors.prometheus_collector import PrometheusCollector
from .collectors.kafka_collector import KafkaCollector
from .collectors.collector_factory import create_collector

# Import modules
from .models.base_model import BaseModel
from .models.prophet_model import ProphetModel
from .models.sklearn_model import SklearnModel
from .models.model_factory import create_model

# Import optimizers
from .optimizers.base_optimizer import BaseOptimizer
from .optimizers.pulp_optimizer import PuLPOptimizer
from .optimizers.optimizer_factory import create_optimizer

# Import exporters
from .exporters.base_exporter import BaseExporter
from .exporters.redis_exporter import RedisExporter
from .exporters.exporter_factory import create_exporter

# Import config
from .config import CACHE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveAnalyzer:
    """Main class for predictive analytics and resource optimization"""
    
    def __init__(self, 
                 data_collector: BaseCollector,
                 model_cache_path: str = "models/cached_models",
                 prediction_horizon: int = 30):  # minutes
        """
        Initialize the PredictiveAnalyzer
        
        Args:
            data_collector: DataCollector instance for metrics collection
            model_cache_path: Path to cache trained models
            prediction_horizon: Number of minutes to forecast into the future
        """
        self.data_collector = data_collector
        self.model_cache_path = model_cache_path
        self.prediction_horizon = prediction_horizon
        self.prophet_model = None
        self.sklearn_model = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(model_cache_path, exist_ok=True)
    
    def collect_metrics(self, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect metrics from the configured data source
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            DataFrame containing collected metrics
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
            
        try:
            data = self.data_collector.collect_metrics(start_time, end_time)
            return self._preprocess_data(data)
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the collected data
        
        Args:
            data: Raw collected data
            
        Returns:
            Preprocessed DataFrame
        """
        if data.empty:
            logger.warning("Empty dataframe provided for preprocessing")
            return data
            
        try:
            # 创建一个副本，避免修改原始数据
            processed_data = data.copy()
            
            # 1. 处理缺失值
            # 检查每个指标的缺失值
            missing_values = processed_data.isnull().sum()
            if missing_values.any():
                logger.info(f"Handling missing values: {missing_values}")
                
                # 对时间序列数据，使用前向填充，然后后向填充
                processed_data = processed_data.fillna(method='ffill')
                processed_data = processed_data.fillna(method='bfill')
                
                # 如果仍有缺失值，使用列的均值填充
                if processed_data.isnull().any().any():
                    for col in processed_data.columns:
                        if col != 'timestamp' and processed_data[col].isnull().any():
                            col_mean = processed_data[col].mean()
                            processed_data[col] = processed_data[col].fillna(col_mean)
                            logger.info(f"Filled missing values in {col} with mean: {col_mean}")
            
            # 2. 确保时间戳是日期时间格式
            if 'timestamp' in processed_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(processed_data['timestamp']):
                    processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                # 将timestamp设为索引，便于时间序列操作
                processed_data = processed_data.set_index('timestamp').sort_index()
            
            # 3. 异常值处理（使用3个标准差）
            for col in processed_data.columns:
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    mean = processed_data[col].mean()
                    std = processed_data[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    
                    # 找到异常值
                    outliers = processed_data[(processed_data[col] < lower_bound) | 
                                            (processed_data[col] > upper_bound)].index
                    
                    if len(outliers) > 0:
                        logger.info(f"Found {len(outliers)} outliers in {col}, capping values")
                        # 使用上下限替代异常值
                        processed_data.loc[processed_data[col] < lower_bound, col] = lower_bound
                        processed_data.loc[processed_data[col] > upper_bound, col] = upper_bound
            
            # 4. 特征工程
            # 添加时间特征
            if isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data['hour'] = processed_data.index.hour
                processed_data['day_of_week'] = processed_data.index.dayofweek
                processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)
                processed_data['month'] = processed_data.index.month
                processed_data['day'] = processed_data.index.day
                
                # 添加滞后特征（过去1小时和过去24小时的均值）
                for col in processed_data.columns:
                    if col not in ['hour', 'day_of_week', 'is_weekend', 'month', 'day'] and pd.api.types.is_numeric_dtype(processed_data[col]):
                        # 1小时滞后均值 (假设数据是5分钟间隔)
                        processed_data[f'{col}_lag_1h'] = processed_data[col].rolling(window=12).mean()
                        # 24小时滞后均值
                        processed_data[f'{col}_lag_24h'] = processed_data[col].rolling(window=288).mean()
            
            # 5. 数据标准化/归一化
            # 对特征列应用Min-Max归一化
            numeric_cols = [col for col in processed_data.columns 
                           if pd.api.types.is_numeric_dtype(processed_data[col]) 
                           and col not in ['hour', 'day_of_week', 'is_weekend', 'month', 'day']]
            
            for col in numeric_cols:
                min_val = processed_data[col].min()
                max_val = processed_data[col].max()
                if max_val > min_val:  # 避免除以零
                    processed_data[f'{col}_normalized'] = (processed_data[col] - min_val) / (max_val - min_val)
            
            # 6. 重置时间戳索引，转回普通列
            if isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data = processed_data.reset_index()
            
            # 7. 处理剩余的缺失值（可能由特征工程引入）
            if processed_data.isnull().any().any():
                processed_data = processed_data.fillna(0)
                logger.info("Filled remaining NaN values with 0")
            
            logger.info(f"Preprocessing complete. Original shape: {data.shape}, New shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            # 如果处理失败，返回原始数据
        return data
    
    def predict_resources(self, 
                         data: pd.DataFrame,
                         use_prophet: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions for resource usage
        
        Args:
            data: Input data for prediction
            use_prophet: Whether to use Prophet or sklearn
            
        Returns:
            Dictionary containing predictions for each metric
        """
        try:
            if use_prophet:
                return self._predict_with_prophet(data)
            return self._predict_with_sklearn(data)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _predict_with_prophet(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using Prophet
        
        Args:
            data: Input DataFrame with timestamp and metric columns
            
        Returns:
            Dictionary containing predictions for each metric
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Input data must contain a 'timestamp' column")
            
        # 获取需要预测的指标列
        metric_columns = [col for col in data.columns 
                        if col not in ['timestamp', 'hour', 'day_of_week', 'is_weekend', 'month', 'day'] 
                        and not col.endswith('_normalized')
                        and not col.endswith('_lag_1h')
                        and not col.endswith('_lag_24h')]
        
        if not metric_columns:
            raise ValueError("No valid metric columns found in the input data")
            
        predictions = {}
        
        for metric in metric_columns:
            try:
                # 准备Prophet格式的数据
                prophet_data = pd.DataFrame({
                    'ds': pd.to_datetime(data['timestamp']),
                    'y': data[metric]
                })
                
                # 检查是否已存在缓存的模型
                model_name = f"prophet_{metric}"
                model = self._load_cached_model(model_name)
                
                if model is None:
                    # 创建新模型
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0,
                        interval_width=0.95
                    )
                    
                    # 添加节假日因素（如果需要）
                    # model.add_country_holidays(country_name='US')
                    
                    # 添加自定义季节性
                    model.add_seasonality(name='hourly', period=24, fourier_order=8)
                    
                    # 训练模型
                    model.fit(prophet_data)
                    
                    # 缓存模型
                    self._cache_model(model, model_name)
                    logger.info(f"Trained and cached new Prophet model for {metric}")
                else:
                    logger.info(f"Loaded cached Prophet model for {metric}")
                
                # 创建未来数据点
                future = model.make_future_dataframe(
                    periods=self.prediction_horizon,
                    freq='min',
                    include_history=False
                )
                
                # 进行预测
                forecast = model.predict(future)
                
                # 提取预测结果
                predictions[metric] = forecast['yhat'].values
                predictions[f'{metric}_lower'] = forecast['yhat_lower'].values
                predictions[f'{metric}_upper'] = forecast['yhat_upper'].values
                
                # 存储预测时间戳
                if 'timestamp' not in predictions:
                    predictions['timestamp'] = forecast['ds'].values
                
                logger.info(f"Prediction for {metric} complete with horizon {self.prediction_horizon}")
                
            except Exception as e:
                logger.error(f"Error predicting {metric} with Prophet: {str(e)}")
                raise RuntimeError(f"Failed to predict {metric}: {str(e)}")
        
        return predictions
    
    def _predict_with_sklearn(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using scikit-learn
        
        Args:
            data: Input DataFrame with timestamp and metric columns
            
        Returns:
            Dictionary containing predictions for each metric
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Input data must contain a 'timestamp' column")
            
        # 时间戳转换为datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # 获取目标指标列（不包括衍生特征和时间特征）
        target_columns = [col for col in data.columns 
                         if col not in ['timestamp', 'hour', 'day_of_week', 'is_weekend', 'month', 'day'] 
                         and not col.endswith('_normalized')
                         and not col.endswith('_lag_1h')
                         and not col.endswith('_lag_24h')]
        
        # 创建特征列表
        feature_columns = [col for col in data.columns 
                          if col in ['hour', 'day_of_week', 'is_weekend', 'month', 'day'] 
                          or col.endswith('_lag_1h') 
                          or col.endswith('_lag_24h')]
        
        # 如果没有足够的特征，尝试创建一些基本特征
        if len(feature_columns) < 3:
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            feature_columns = ['hour', 'day_of_week', 'month']
        
        # 准备预测结果字典
        predictions = {}
        
        # 创建未来时间点
        last_timestamp = data['timestamp'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp,
            periods=self.prediction_horizon + 1,
            freq='min'
        )[1:]  # 排除当前时间点
        
        # 创建未来的特征数据
        future_data = pd.DataFrame({'timestamp': future_timestamps})
        future_data['hour'] = future_data['timestamp'].dt.hour
        future_data['day_of_week'] = future_data['timestamp'].dt.dayofweek
        future_data['is_weekend'] = future_data['day_of_week'].isin([5, 6]).astype(int)
        future_data['month'] = future_data['timestamp'].dt.month
        future_data['day'] = future_data['timestamp'].dt.day
        
        # 存储时间戳
        predictions['timestamp'] = future_timestamps.values
        
        # 对每个目标指标进行预测
        for target in target_columns:
            try:
                # 检查是否有缓存的模型
                model_name = f"sklearn_{target}"
                model = self._load_cached_model(model_name)
                
                if model is None:
                    # 创建新模型
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=10,
                        random_state=42
                    )
                    
                    # 准备训练数据（移除包含NaN的行）
                    train_data = data.dropna(subset=feature_columns + [target])
                    
                    if len(train_data) < 10:
                        logger.warning(f"Not enough data to train model for {target}. Using mean value.")
                        predictions[target] = np.full(self.prediction_horizon, data[target].mean())
                        continue
                    
                    # 训练模型
                    X_train = train_data[feature_columns]
                    y_train = train_data[target]
                    model.fit(X_train, y_train)
                    
                    # 缓存模型
                    self._cache_model(model, model_name)
                    logger.info(f"Trained and cached new sklearn model for {target}")
                else:
                    logger.info(f"Loaded cached sklearn model for {target}")
                
                # 准备包含特征的未来数据
                future_features = future_data[feature_columns].copy()
                
                # 处理滞后特征（使用最近的值）
                for col in feature_columns:
                    if col.endswith('_lag_1h') or col.endswith('_lag_24h'):
                        orig_col = col.split('_lag_')[0]
                        # 获取原始特征的最近值
                        if orig_col in data.columns:
                            last_values = data[col].tail(min(24, len(data)))
                            future_features[col] = last_values.mean()
                
                # 预测
                X_future = future_features.fillna(0)  # 确保没有NaN值
                predictions[target] = model.predict(X_future)
                
                # 计算预测区间
                # 使用随机森林的置信区间估计
                predictions[f'{target}_lower'] = predictions[target] * 0.9  # 简单估计下界为90%
                predictions[f'{target}_upper'] = predictions[target] * 1.1  # 简单估计上界为110%
                
                logger.info(f"Prediction for {target} complete with horizon {self.prediction_horizon}")
                
            except Exception as e:
                logger.error(f"Error predicting {target} with sklearn: {str(e)}")
                # 如果某个特定目标的预测失败，继续处理其他目标
                predictions[target] = np.full(self.prediction_horizon, data[target].mean())
                predictions[f'{target}_lower'] = predictions[target] * 0.9
                predictions[f'{target}_upper'] = predictions[target] * 1.1
        
        return predictions
    
    def optimize_allocation(self, 
                          predictions: Dict[str, np.ndarray],
                          constraints: Dict) -> Dict[str, float]:
        """
        Optimize resource allocation based on predictions
        
        Args:
            predictions: Dictionary of predictions for each metric
            constraints: Resource constraints
            
        Returns:
            Optimized resource allocation plan
        """
        try:
            # 验证预测和约束
            if not predictions:
                raise ValueError("Empty predictions")
            
            required_metrics = {'cpu_usage', 'memory_usage', 'network_io'}
            if not all(metric in predictions for metric in required_metrics):
                logger.warning(f"Missing required metrics in predictions. Required: {required_metrics}")
                # 为缺失的指标使用默认值
                for metric in required_metrics:
                    if metric not in predictions:
                        predictions[metric] = np.array([constraints.get(f'max_{metric}', 100) * 0.5])
                        logger.info(f"Using default value for {metric}: {predictions[metric][0]}")
            
            required_constraints = {'max_cpu', 'max_memory', 'max_network', 'max_latency'}
            if not all(const in constraints for const in required_constraints):
                raise ValueError(f"Missing required constraints. Required: {required_constraints}")
            
            # 获取预测的最大值（作为资源需求）
            cpu_demand = np.max(predictions['cpu_usage'])
            memory_demand = np.max(predictions['memory_usage'])
            network_demand = np.max(predictions['network_io'])
            
            # 如果有延迟指标，也处理
            latency_demand = 1.0  # 默认值
            if 'latency' in predictions:
                latency_demand = np.max(predictions['latency'])
            
            # 创建PuLP优化问题
            prob = LpProblem("Resource_Allocation", LpMinimize)
            
            # 决策变量（资源分配）
            cpu_alloc = LpVariable("cpu_allocation", 0, constraints['max_cpu'])
            memory_alloc = LpVariable("memory_allocation", 0, constraints['max_memory'])
            network_alloc = LpVariable("network_allocation", 0, constraints['max_network'])
            
            # 目标函数权重
            cpu_weight = 0.4
            memory_weight = 0.3
            network_weight = 0.3
            
            # 设置目标函数（最小化成本或最大化性能）
            # 默认为最小化成本
            objective = (
                cpu_weight * cpu_alloc / constraints['max_cpu'] + 
                memory_weight * memory_alloc / constraints['max_memory'] + 
                network_weight * network_alloc / constraints['max_network']
            )
            
            # 如果目标是性能优化，则反转目标函数
            if constraints.get('optimization_objective') == 'performance':
                objective = -objective  # 最大化性能
                
            prob += objective
            
            # 添加约束
            # 1. 资源分配必须满足预测需求，加上安全边际
            safety_margin = constraints.get('safety_margin', 1.2)  # 默认安全边际为20%
            prob += cpu_alloc >= cpu_demand * safety_margin
            prob += memory_alloc >= memory_demand * safety_margin
            prob += network_alloc >= network_demand * safety_margin
            
            # 2. 延迟约束（通过资源分配影响）
            max_allowed_latency = constraints['max_latency']
            latency_factor = min(latency_demand / max_allowed_latency, 1.0)
            
            # 如果预测延迟接近最大允许值，增加资源分配
            if latency_factor > 0.8:
                latency_scaling = 1 + (latency_factor - 0.8) * 2  # 0.8->1.0, 1.0->1.4
                prob += cpu_alloc >= cpu_demand * safety_margin * latency_scaling
                prob += memory_alloc >= memory_demand * safety_margin * latency_scaling
            
            # 3. 资源平衡约束
            if constraints.get('balance_resources', True):
                # 创建辅助变量来实现平衡
                balance_cpu_memory = LpVariable("balance_cpu_memory", 0)
                balance_memory_network = LpVariable("balance_memory_network", 0)
                
                # 标准化资源值，确保可比较
                cpu_norm = cpu_alloc / constraints['max_cpu']
                memory_norm = memory_alloc / constraints['max_memory']
                network_norm = network_alloc / constraints['max_network']
                
                # 添加平衡约束
                balance_tolerance = constraints.get('balance_tolerance', 0.2)
                
                # |cpu_norm - memory_norm| <= balance_tolerance
                prob += cpu_norm - memory_norm <= balance_tolerance
                prob += memory_norm - cpu_norm <= balance_tolerance
                
                # |memory_norm - network_norm| <= balance_tolerance
                prob += memory_norm - network_norm <= balance_tolerance
                prob += network_norm - memory_norm <= balance_tolerance
            
            # 最小利用率约束
            min_utilization = constraints.get('min_utilization', 20) / 100  # 百分比转小数
            prob += cpu_alloc >= constraints['max_cpu'] * min_utilization
            prob += memory_alloc >= constraints['max_memory'] * min_utilization
            prob += network_alloc >= constraints['max_network'] * min_utilization
            
            # 求解优化问题
            solver_name = constraints.get('solver', 'CBC')
            solver = None
            
            # 设置求解器
            if solver_name == 'CBC':
                solver = PULP_CBC_CMD(msg=False, timeLimit=constraints.get('solution_timeout', 60))
            elif solver_name == 'GLPK':
                solver = GLPK_CMD(msg=False, timeLimit=constraints.get('solution_timeout', 60))
            else:
                # 默认使用CBC
                solver = PULP_CBC_CMD(msg=False)
            
            # 求解
            status = prob.solve(solver)
            
            # 检查求解状态
            if LpStatus[status] != 'Optimal':
                logger.warning(f"Non-optimal solution: {LpStatus[status]}")
                # 如果优化失败，返回安全分配（需求 * 安全边际）
                return {
                    'cpu_allocation': min(cpu_demand * safety_margin, constraints['max_cpu']),
                    'memory_allocation': min(memory_demand * safety_margin, constraints['max_memory']),
                    'network_allocation': min(network_demand * safety_margin, constraints['max_network']),
                    'status': 'fallback',
                    'solver_status': LpStatus[status],
                    'objective_value': None
                }
            
            # 获取优化结果
            result = {
                'cpu_allocation': value(cpu_alloc),
                'memory_allocation': value(memory_alloc),
                'network_allocation': value(network_alloc),
                'status': 'optimal',
                'solver_status': LpStatus[status],
                'objective_value': value(prob.objective)
            }
            
            # 计算资源利用率
            result['utilization'] = {
                'cpu': cpu_demand / result['cpu_allocation'] * 100 if result['cpu_allocation'] > 0 else 0,
                'memory': memory_demand / result['memory_allocation'] * 100 if result['memory_allocation'] > 0 else 0,
                'network': network_demand / result['network_allocation'] * 100 if result['network_allocation'] > 0 else 0
            }
            
            logger.info(f"Resource optimization completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {str(e)}")
            # 发生错误时返回保守的资源分配
            safety_margin = constraints.get('safety_margin', 1.2)
            return {
                'cpu_allocation': constraints['max_cpu'] * 0.5,
                'memory_allocation': constraints['max_memory'] * 0.5,
                'network_allocation': constraints['max_network'] * 0.5,
                'status': 'error',
                'error_message': str(e),
                'utilization': {
                    'cpu': 50,
                    'memory': 50,
                    'network': 50
                }
            }
    
    def export_results(self,
                      predictions: Dict[str, np.ndarray],
                      optimization_results: Dict[str, float],
                      export_format: str = "json") -> Dict:
        """
        Export prediction and optimization results
        
        Args:
            predictions: Prediction results
            optimization_results: Optimization results
            export_format: Desired export format
            
        Returns:
            Formatted results
        """
        try:
            # 验证输入
            if not predictions:
                raise ValueError("Empty predictions")
            if not optimization_results:
                raise ValueError("Empty optimization results")
                
            # 准备导出结果
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'predictions': {},
                'optimization': optimization_results,
                'metadata': {
                    'export_format': export_format,
                    'version': '1.0.0'
                }
            }
            
            # 处理预测结果
            # 如果predictions包含numpy数组，转换为列表
            for key, value in predictions.items():
                if isinstance(value, np.ndarray):
                    # 将timestamp转换为ISO格式字符串
                    if key == 'timestamp':
                        if isinstance(value[0], np.datetime64):
                            # 转换numpy datetime64数组为ISO格式字符串列表
                            export_data['predictions'][key] = [pd.Timestamp(ts).isoformat() for ts in value]
                        else:
                            # 尝试从时间戳值创建日期时间
                            export_data['predictions'][key] = [
                                (datetime.now() + timedelta(minutes=i)).isoformat() 
                                for i in range(len(value))
                            ]
                    else:
                        # 其他数值转换为列表
                        export_data['predictions'][key] = value.tolist()
                else:
                    export_data['predictions'][key] = value
            
            # 确保结果包含时间戳
            if 'timestamp' not in export_data['predictions']:
                # 创建一个默认的时间序列
                prediction_len = len(next(iter(export_data['predictions'].values())))
                export_data['predictions']['timestamp'] = [
                    (datetime.now() + timedelta(minutes=i)).isoformat() 
                    for i in range(prediction_len)
                ]
            
            # 转换为标准格式
            if export_format.lower() == 'json':
                # 直接返回处理后的字典，调用者可以使用json.dumps进行序列化
                return export_data
                
            elif export_format.lower() == 'csv':
                # 将预测结果转换为DataFrame
                prediction_df = pd.DataFrame()
                
                # 处理时间戳
                if 'timestamp' in export_data['predictions']:
                    prediction_df['timestamp'] = export_data['predictions']['timestamp']
                
                # 添加其他预测列
                for key, values in export_data['predictions'].items():
                    if key != 'timestamp':
                        prediction_df[key] = values
                
                # 添加优化结果列（会重复相同的值）
                for key, value in export_data['optimization'].items():
                    if key not in ['status', 'solver_status', 'utilization']:
                        prediction_df[f'opt_{key}'] = value
                
                # 将DataFrame转换为CSV字符串
                csv_data = prediction_df.to_csv(index=False)
                
                # 返回包含CSV数据的字典
                return {
                    'format': 'csv',
                    'data': csv_data,
                    'timestamp': export_data['timestamp'],
                    'metadata': export_data['metadata']
                }
                
            elif export_format.lower() == 'redis':
                # 使用配置的Redis信息
                from redis import Redis
                
                # 连接Redis
                redis_client = Redis(
                    host=CACHE_CONFIG.get('host', 'localhost'),
                    port=CACHE_CONFIG.get('port', 6379),
                    db=CACHE_CONFIG.get('db', 0)
                )
                
                # 生成键
                timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
                key = f"{CACHE_CONFIG.get('key_prefix', 'prediction:')}_{timestamp_str}"
                
                # 将数据序列化为JSON并保存到Redis
                import json
                redis_client.set(
                    key, 
                    json.dumps(export_data),
                    ex=CACHE_CONFIG.get('expire', 3600)
                )
                
                # 更新最新键
                redis_client.set(
                    f"{CACHE_CONFIG.get('key_prefix', 'prediction:')}latest",
                    key,
                    ex=CACHE_CONFIG.get('expire', 3600)
                )
                
                # 返回结果
                return {
                    'format': 'redis',
                    'key': key,
                    'timestamp': export_data['timestamp'],
                    'expiry': CACHE_CONFIG.get('expire', 3600),
                    'metadata': export_data['metadata']
                }
                
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            # 发生错误时，返回错误信息
            return {
                'error': True,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _cache_model(self, model: object, model_name: str):
        """Cache a trained model"""
        try:
            joblib.dump(model, f"{self.model_cache_path}/{model_name}.joblib")
        except Exception as e:
            logger.warning(f"Failed to cache model: {str(e)}")
    
    def _load_cached_model(self, model_name: str) -> Optional[object]:
        """Load a cached model if available"""
        try:
            return joblib.load(f"{self.model_cache_path}/{model_name}.joblib")
        except:
            return None 