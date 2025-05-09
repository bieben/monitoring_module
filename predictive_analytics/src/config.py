"""Configuration settings for the Predictive Analytics Module"""
import os

# 获取环境变量或使用默认值
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

# 调整日志级别
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Data Collection Settings
PROMETHEUS_CONFIG = {
    'url': f"{PROMETHEUS_URL}/api/v1/query_range",
    'metrics': {
        'requests_total': 'model_service_requests_total{endpoint="/predict"}',
        'latency_avg': 'rate(model_service_response_time_seconds_sum{endpoint="/predict"}[5m]) / rate(model_service_response_time_seconds_count{endpoint="/predict"}[5m])',
        'latency_p95': 'histogram_quantile(0.95, sum(rate(model_service_response_time_seconds_bucket{endpoint="/predict"}[5m])) by (le, model_id))',
        'latency_p99': 'histogram_quantile(0.99, sum(rate(model_service_response_time_seconds_bucket{endpoint="/predict"}[5m])) by (le, model_id))'
    },
    'query_timeout': 30,
    'retry_attempts': 3,
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'requests_total': (0, 1000000),
            'latency_avg': (0, 100),
            'latency_p95': (0, 100),
            'latency_p99': (0, 100)
        }
    }
}

KAFKA_CONFIG = {
    'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
    'topics': {
        'metrics': {
            'name': 'metrics',
            'partitions': 3,
            'replication_factor': 1
        }
    },
    'group_id': 'predictive_analytics',
    'consumer_config': {
        'auto_offset_reset': 'earliest',
        'enable_auto_commit': True,
        'max_poll_interval_ms': 300000,
        'max_poll_records': 500
    },
    'schema': {
        'metric_name': 'string',
        'value': 'float',
        'timestamp': 'string',
        'labels': 'dict'
    },
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'cpu_usage': (0, 100),
            'memory_usage': (0, 32 * 1024 * 1024 * 1024),
            'network_io': (0, 10 * 1024 * 1024 * 1024),
            'latency': (0, 100)
        }
    }
}

# Model Settings
PROPHET_CONFIG = {
    'model_params': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'interval_width': 0.95,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'return_intervals': True
    },
    'validation': {
        'initial_training_period': '7d',
        'horizon': '1d',
        'cross_validation_windows': 3,
        'metrics': ['rmse', 'mae', 'mape']
    },
    'features': ['timestamp', 'requests_total', 'latency_avg', 'latency_p95', 'latency_p99'],
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'requests_total': (0, 1000000),
            'latency_avg': (0, 100),
            'latency_p95': (0, 10),
            'latency_p99': (0, 10)
        }
    }
}

SKLEARN_CONFIG = {
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'validation': {
        'test_size': 0.2,
        'cv_folds': 5,
        'scoring': 'neg_mean_squared_error'
    },
    'features': ['hour', 'day_of_week', 'month', 'requests_total', 'latency_avg', 'latency_p95', 'latency_p99'],
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'requests_total': (0, 1000000),
            'latency_avg': (0, 100),
            'latency_p95': (0, 100),
            'latency_p99': (0, 100)
        }
    }
}

# Resource Optimization Settings
OPTIMIZATION_CONFIG = {
    'objective': 'minimize_cost',
    'constraints': {
        'max_cpu': float(os.environ.get('MAX_CPU', '100')),  # percentage
        'max_memory': float(os.environ.get('MAX_MEMORY', str(32 * 1024 * 1024 * 1024))),  # bytes (32GB)
        'max_network': float(os.environ.get('MAX_NETWORK', str(10 * 1024 * 1024 * 1024))),  # bytes (10GB/s)
        'max_latency': float(os.environ.get('MAX_LATENCY', '1.0'))  # seconds
    },
    'weights': {
        'cpu': 0.4,
        'memory': 0.3,
        'network': 0.3
    },
    'solver_config': {
        'solver': os.environ.get('OPTIMIZER_SOLVER', 'CBC'),
        'safety_margin': float(os.environ.get('SAFETY_MARGIN', '1.2')),
        'balance_resources': os.environ.get('BALANCE_RESOURCES', 'true').lower() == 'true',
        'balance_tolerance': float(os.environ.get('BALANCE_TOLERANCE', '0.2')),
        'min_utilization': float(os.environ.get('MIN_UTILIZATION', '20')),
        'max_utilization': float(os.environ.get('MAX_UTILIZATION', '90')),
        'solution_timeout': int(os.environ.get('SOLUTION_TIMEOUT', '300')),
        'options': {
            'msg': 0,
            'timeLimit': 60
        }
    }
}

# Cache Settings
CACHE_CONFIG = {
    'host': REDIS_HOST,
    'port': REDIS_PORT,
    'db': REDIS_DB,
    'key_prefix': os.environ.get('REDIS_KEY_PREFIX', 'prediction:'),
    'expire': int(os.environ.get('REDIS_EXPIRE', '300'))  # seconds
}

# Logging Settings
LOGGING_CONFIG = {
    'level': LOG_LEVEL,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.environ.get('LOG_FILE', 'predictive_analytics.log'),
            'level': 'DEBUG'
        }
    }
} 