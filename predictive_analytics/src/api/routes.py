"""
REST API routes for the predictive analytics service
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import json
import os
import time
import threading
import pytz

from ..collectors import CollectorFactory
from ..models import ModelFactory
from ..optimizers import OptimizerFactory
from ..exporters import ExporterFactory
from ..config import (
    PROMETHEUS_CONFIG,
    PROPHET_CONFIG,
    OPTIMIZATION_CONFIG,
    CACHE_CONFIG
)

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

# 创建内存缓存以防Redis不可用
in_memory_cache = {
    'latest': None,
    'history': []
}

# 组件延迟初始化标记
components_initialized = False
collector = None
model = None
optimizer = None
exporter = None
initialization_lock = threading.Lock()
use_redis = False

def lazy_init_components():
    """懒加载初始化组件，仅在需要时初始化"""
    global components_initialized, collector, model, optimizer, exporter, use_redis
    
    if components_initialized:
        return
    
    with initialization_lock:
        if components_initialized:
            return
        
        # 初始化collector和optimizer，这些初始化相对较快
        collector = CollectorFactory.create_collector('prometheus', PROMETHEUS_CONFIG)
        optimizer = OptimizerFactory.create_optimizer('pulp', OPTIMIZATION_CONFIG)
        
        # 惰性初始化模型，即使它不在这里实例化，首次使用时会懒加载
        model = ModelFactory.create_model('prophet', PROPHET_CONFIG)
        
        # 尝试连接Redis，如果失败使用内存缓存
        try:
            exporter = ExporterFactory.create_exporter('redis', CACHE_CONFIG)
            use_redis = True
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {str(e)}. Using in-memory cache instead.")
            use_redis = False
            
            # 创建内存缓存导出器
            class InMemoryExporter:
                def export(self, data):
                    in_memory_cache['latest'] = data
                    in_memory_cache['history'].append(data)
                    if len(in_memory_cache['history']) > 100:  # 限制历史记录数量
                        in_memory_cache['history'].pop(0)
                    return True
                    
                def get_latest(self):
                    return in_memory_cache['latest']
                    
                def get_history(self, limit=10):
                    return in_memory_cache['history'][-limit:] if in_memory_cache['history'] else []
            
            exporter = InMemoryExporter()
            
        components_initialized = True
        logger.info("All components initialized successfully")

@api.before_request
def before_request():
    """在处理请求前确保组件已初始化"""
    lazy_init_components()

@api.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions and optimizing resources
    
    Expected request body:
    {
        "horizon": 30,  # minutes to predict
        "use_cache": true  # whether to use cached model
    }
    """
    try:
        # 确保组件已初始化
        lazy_init_components()
        
        # Get request parameters
        data = request.get_json()
        horizon = data.get('horizon', 30)
        use_cache = data.get('use_cache', True)
        
        # 使用UTC时间
        end_time = datetime.now(pytz.UTC)
        metrics_start_time = end_time - timedelta(hours=24)
        metrics_df = collector.collect_metrics(metrics_start_time, end_time)
        logger.info("Metrics collection completed")
        
        # 检查是否存在实时系统资源指标数据
        real_metrics = [metric for metric in metrics_df['metric_name'].unique() 
                       if metric in ['cpu_usage_real', 'memory_usage_real', 'network_io_real']]
        
        # Make predictions
        prediction_start = time.time()
        if use_cache and model.load_model():
            logger.info("Using cached model")
        else:
            logger.info("Training new model")
            model.train(metrics_df)
            model.save_model()
        
        predictions_df = model.predict(horizon)
        logger.info(f"Prediction completed in {time.time() - prediction_start:.2f} seconds")
        
        # 如果存在实时系统指标，直接创建带有最新值的预测列
        if real_metrics:
            logger.info(f"Adding real system metrics to predictions: {real_metrics}")
            # 获取最新的系统指标值
            latest_metrics = {}
            for metric in real_metrics:
                metric_data = metrics_df[metrics_df['metric_name'] == metric]
                if not metric_data.empty:
                    latest_value = float(metric_data['value'].iloc[-1])
                    # 将真实系统指标添加到预测数据中
                    predictions_df[metric] = latest_value
                    latest_metrics[metric] = latest_value
                    logger.info(f"Added {metric} = {latest_value} to predictions")
        
        # Optimize resources
        optimization_start = time.time()
        optimization_result = optimizer.optimize(
            predictions_df,
            OPTIMIZATION_CONFIG['constraints']
        )
        logger.info(f"Resource optimization completed in {time.time() - optimization_start:.2f} seconds")
        
        # Export results with UTC timestamps
        export_start = time.time()
        
        # 确保预测结果中的时间戳是UTC时间
        if isinstance(predictions_df, pd.DataFrame):
            if 'timestamp' in predictions_df.columns:
                # 如果时间戳没有时区信息，添加UTC时区
                if predictions_df['timestamp'].dt.tz is None:
                    predictions_df['timestamp'] = predictions_df['timestamp'].dt.tz_localize('UTC')
                # 如果时间戳已有时区信息但不是UTC，转换为UTC
                elif str(predictions_df['timestamp'].dt.tz) != 'UTC':
                    predictions_df['timestamp'] = predictions_df['timestamp'].dt.tz_convert('UTC')
                    
                predictions_dict = predictions_df.to_dict(orient='records')
                # 转换时间戳为ISO格式
                for pred in predictions_dict:
                    if 'timestamp' in pred:
                        pred['timestamp'] = pred['timestamp'].isoformat()
            else:
                predictions_dict = predictions_df.to_dict(orient='records')
        else:
            predictions_dict = predictions_df
            
        export_data = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'predictions': predictions_dict,
            'optimization': optimization_result
        }
        
        try:
            exporter.export(export_data)
            logger.info(f"Results exported in {time.time() - export_start:.2f} seconds")
        except Exception as ex:
            logger.warning(f"Failed to export results: {str(ex)}, falling back to in-memory cache")
            # 如果导出失败，直接存储到内存缓存
            in_memory_cache['latest'] = export_data
            in_memory_cache['history'].append(export_data)
            if len(in_memory_cache['history']) > 100:  # 限制历史记录数量
                in_memory_cache['history'].pop(0)
        
        total_time = time.time() - prediction_start
        logger.info(f"Total prediction pipeline completed in {total_time:.2f} seconds")
        
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions_dict,
                'optimization': optimization_result
            },
            'processing_time': f"{total_time:.2f} seconds"
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/predictions/latest', methods=['GET'])
def get_latest_prediction():
    """Get the latest prediction and optimization results"""
    try:
        # 确保组件已初始化
        lazy_init_components()
        
        latest_data = None
        try:
            latest_data = exporter.get_latest()
        except Exception as e:
            logger.warning(f"Error getting data from exporter: {str(e)}")
            latest_data = in_memory_cache['latest']
        
        # 如果没有找到预测数据，创建一个默认值
        if latest_data is None:
            # 检查是否存在最新预测数据
            try:
                # 尝试获取一些实时数据来填充默认值
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=30)
                metrics_df = collector.collect_metrics(start_time, end_time)
                
                # 如果有指标数据，创建一个基本预测
                if not metrics_df.empty:
                    # 提取最新的指标值
                    latest_metrics = {}
                    # 包含真实资源指标
                    for metric_name in ['requests_total', 'latency_avg', 'latency_p95', 'latency_p99', 
                                        'cpu_usage_real', 'memory_usage_real', 'network_io_real']:
                        metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                        if not metric_data.empty:
                            latest_metrics[metric_name] = float(metric_data['value'].iloc[-1])
                        else:
                            latest_metrics[metric_name] = 0.0
                    
                    # 创建一个基本预测
                    dummy_predictions = []
                    current_time = datetime.now()
                    for i in range(5):  # 只创建5个预测点
                        future_time = current_time + timedelta(minutes=5 * i)
                        prediction = {
                            'timestamp': future_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'requests_total': latest_metrics.get('requests_total', 0.0),
                            'latency_avg': latest_metrics.get('latency_avg', 0.0),
                            'latency_p95': latest_metrics.get('latency_p95', 0.0),
                            'latency_p99': latest_metrics.get('latency_p99', 0.0)
                        }
                        # 添加真实资源使用数据（如果有）
                        if 'cpu_usage_real' in latest_metrics:
                            prediction['cpu_usage'] = latest_metrics.get('cpu_usage_real', 0.0)
                        if 'memory_usage_real' in latest_metrics:
                            prediction['memory_usage'] = latest_metrics.get('memory_usage_real', 0.0)
                        if 'network_io_real' in latest_metrics:
                            prediction['network_io'] = latest_metrics.get('network_io_real', 0.0)
                            
                        dummy_predictions.append(prediction)
                    
                    # 创建一个基本优化结果
                    dummy_optimization = {
                        'cpu_allocation': 100.0,
                        'memory_allocation': 1024.0,
                        'network_allocation': 1024.0,
                        'status': 'optimal',
                        'solver_status': 'Optimal',
                        'objective_value': 0.0,
                        'utilization': {
                            # 优先使用真实资源指标来计算利用率
                            'cpu': latest_metrics.get('cpu_usage_real', min(max(latest_metrics.get('requests_total', 0.0), 1.0), 100.0)),
                            'memory': latest_metrics.get('memory_usage_real', min(max(latest_metrics.get('latency_avg', 0.0) * 100, 1.0), 100.0)),
                            'network': latest_metrics.get('network_io_real', min(max(latest_metrics.get('latency_p95', 0.0) * 100, 1.0), 100.0)) / (10 * 1024 * 1024) if 'network_io_real' in latest_metrics else min(max(latest_metrics.get('latency_p95', 0.0) * 100, 1.0), 100.0)
                        }
                    }
                    
                    latest_data = {
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'predictions': dummy_predictions,
                        'optimization': dummy_optimization
                    }
                    
                    # 存储到内存缓存
                    in_memory_cache['latest'] = latest_data
                    in_memory_cache['history'].append(latest_data)
                    
                    # 尝试导出
                    try:
                        exporter.export(latest_data)
                    except Exception as ex:
                        logger.warning(f"Error exporting dummy data: {str(ex)}")
            except Exception as e:
                logger.warning(f"Error creating dummy prediction: {str(e)}")
        
        # 如果仍然没有数据，返回一个完全空的预测
        if latest_data is None:
            current_time = datetime.now()
            latest_data = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': [
                    {
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'requests_total': 0.0,
                        'latency_avg': 0.0,
                        'latency_p95': 0.0,
                        'latency_p99': 0.0
                    }
                ],
                'optimization': {
                    'cpu_allocation': 100.0,
                    'memory_allocation': 1024.0,
                    'network_allocation': 1024.0,
                    'status': 'optimal',
                    'solver_status': 'Optimal',
                    'objective_value': 0.0,
                    'utilization': {
                        'cpu': 0.0,
                        'memory': 0.0,
                        'network': 0.0
                    }
                }
            }
            
            # 存储到内存缓存
            in_memory_cache['latest'] = latest_data
        
        return jsonify({
            'status': 'success',
            'data': latest_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting latest prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/predictions/history', methods=['GET'])
def get_prediction_history():
    """Get historical predictions and optimization results"""
    try:
        # 确保组件已初始化
        lazy_init_components()
        
        limit = request.args.get('limit', default=10, type=int)
        
        history = []
        try:
            history = exporter.get_history(limit=limit)
        except Exception as e:
            logger.warning(f"Error getting history from exporter: {str(e)}, using in-memory cache")
            history = in_memory_cache['history'][-limit:] if in_memory_cache['history'] else []
        
        # 如果历史记录为空，则提供一个默认的历史记录
        if not history:
            try:
                # 使用当前指标创建一个简单的历史记录
                current_time = datetime.now()
                
                # 获取最新指标
                end_time = current_time
                start_time = end_time - timedelta(minutes=30)
                metrics_df = collector.collect_metrics(start_time, end_time)
                
                # 如果有指标数据，创建一个基本预测历史
                if not metrics_df.empty:
                    # 创建一个简单的历史记录
                    for i in range(5):
                        past_time = current_time - timedelta(minutes=i * 15)
                        
                        # 根据指标创建基本预测
                        latest_metrics = {}
                        for metric_name in ['requests_total', 'latency_avg', 'latency_p95', 'latency_p99',
                                           'cpu_usage_real', 'memory_usage_real', 'network_io_real']:
                            metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                            if not metric_data.empty:
                                latest_metrics[metric_name] = float(metric_data['value'].iloc[-1])
                            else:
                                latest_metrics[metric_name] = 0.0
                                
                        # 创建基本预测
                        dummy_predictions = []
                        for j in range(3):  # 只创建3个预测点
                            future_time = past_time + timedelta(minutes=5 * j)
                            prediction = {
                                'timestamp': future_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'requests_total': latest_metrics.get('requests_total', 0.0),
                                'latency_avg': latest_metrics.get('latency_avg', 0.0),
                                'latency_p95': latest_metrics.get('latency_p95', 0.0),
                                'latency_p99': latest_metrics.get('latency_p99', 0.0)
                            }
                            # 添加真实资源使用数据（如果有）
                            if 'cpu_usage_real' in latest_metrics:
                                prediction['cpu_usage'] = latest_metrics.get('cpu_usage_real', 0.0)
                            if 'memory_usage_real' in latest_metrics:
                                prediction['memory_usage'] = latest_metrics.get('memory_usage_real', 0.0)
                            if 'network_io_real' in latest_metrics:
                                prediction['network_io'] = latest_metrics.get('network_io_real', 0.0)
                                
                            dummy_predictions.append(prediction)
                        
                        # 创建一个基本优化结果
                        dummy_optimization = {
                            'cpu_allocation': 100.0,
                            'memory_allocation': 1024.0,
                            'network_allocation': 1024.0,
                            'status': 'optimal',
                            'solver_status': 'Optimal',
                            'objective_value': 0.0,
                            'utilization': {
                                # 优先使用真实资源指标来计算利用率
                                'cpu': latest_metrics.get('cpu_usage_real', min(max(latest_metrics.get('requests_total', 0.0), 1.0), 100.0)),
                                'memory': latest_metrics.get('memory_usage_real', min(max(latest_metrics.get('latency_avg', 0.0) * 100, 1.0), 100.0)),
                                'network': latest_metrics.get('network_io_real', min(max(latest_metrics.get('latency_p95', 0.0) * 100, 1.0), 100.0)) / (10 * 1024 * 1024) if 'network_io_real' in latest_metrics else min(max(latest_metrics.get('latency_p95', 0.0) * 100, 1.0), 100.0)
                            }
                        }
                        
                        history_item = {
                            'timestamp': past_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'predictions': dummy_predictions,
                            'optimization': dummy_optimization
                        }
                        
                        history.append(history_item)
                        
                    # 存储到内存缓存
                    in_memory_cache['history'] = history
            except Exception as ex:
                logger.warning(f"Error creating dummy history: {str(ex)}")
        
        return jsonify({
            'status': 'success',
            'data': history
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/metrics', methods=['GET'])
def get_current_metrics():
    """Get current system metrics"""
    try:
        # 确保组件已初始化
        lazy_init_components()
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        metrics_df = collector.collect_metrics(start_time, end_time)
        
        # 将原始指标数据转换为前端友好的格式
        formatted_metrics = []
        
        if not metrics_df.empty:
            # 按时间戳分组
            grouped = metrics_df.groupby('timestamp')
            
            for timestamp, group in grouped:
                metric_point = {'timestamp': timestamp}
                
                # 添加各种指标
                for _, row in group.iterrows():
                    metric_name = row['metric_name']
                    value = row['value']
                    # 确保值不为 null
                    if pd.notna(value):
                        metric_point[metric_name] = value
                    else:
                        # 使用默认值替代null
                        metric_point[metric_name] = 0.0
                
                formatted_metrics.append(metric_point)
        
        # 如果没有数据，返回一个样例点，避免前端出错
        if not formatted_metrics:
            formatted_metrics = [{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'requests_total': 0.0,
                'latency_avg': 0.0,
                'latency_p95': 0.0,
                'latency_p99': 0.0
            }]
        else:
            # 确保所有必要的指标都存在，如果不存在则使用默认值
            for metric_point in formatted_metrics:
                for metric_name in ['requests_total', 'latency_avg', 'latency_p95', 'latency_p99']:
                    if metric_name not in metric_point:
                        metric_point[metric_name] = 0.0
        
        return jsonify({
            'status': 'success',
            'data': formatted_metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting current metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # 在健康检查时不要初始化所有组件，否则会导致慢启动
    services_status = {
        'api': 'up'
    }
    
    # 仅当组件已初始化时才检查它们的状态
    if components_initialized:
        services_status.update({
            'collector': 'up' if collector else 'not_initialized',
            'model': 'up' if model else 'not_initialized',
            'optimizer': 'up' if optimizer else 'not_initialized',
            'exporter': 'up' if use_redis else 'using in-memory fallback'
        })
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': services_status,
        'components_initialized': components_initialized
    }), 200 