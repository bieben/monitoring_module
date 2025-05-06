"""
REST API routes for the predictive analytics service
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

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

# Initialize components
collector = CollectorFactory.create_collector('prometheus', PROMETHEUS_CONFIG)
model = ModelFactory.create_model('prophet', PROPHET_CONFIG)
optimizer = OptimizerFactory.create_optimizer('pulp', OPTIMIZATION_CONFIG)
exporter = ExporterFactory.create_exporter('redis', CACHE_CONFIG)

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
        # Get request parameters
        data = request.get_json()
        horizon = data.get('horizon', 30)
        use_cache = data.get('use_cache', True)
        
        # Collect metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        metrics_df = collector.collect_metrics(start_time, end_time)
        
        # Make predictions
        if use_cache and model.load_model():
            logger.info("Using cached model")
        else:
            logger.info("Training new model")
            model.train(metrics_df)
            model.save_model()
        
        predictions_df = model.predict(horizon)
        
        # Optimize resources
        optimization_result = optimizer.optimize(
            predictions_df,
            OPTIMIZATION_CONFIG['constraints']
        )
        
        # Export results
        export_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': predictions_df,
            'optimization': optimization_result
        }
        exporter.export(export_data)
        
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions_df.to_dict(orient='records'),
                'optimization': optimization_result
            }
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
        latest_data = exporter.get_latest()
        
        if latest_data is None:
            return jsonify({
                'status': 'error',
                'message': 'No predictions available'
            }), 404
            
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
        limit = request.args.get('limit', default=10, type=int)
        history = exporter.get_history(limit=limit)
        
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
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        metrics_df = collector.collect_metrics(start_time, end_time)
        
        return jsonify({
            'status': 'success',
            'data': metrics_df.to_dict(orient='records')
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200 