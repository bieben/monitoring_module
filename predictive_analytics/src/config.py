"""Configuration settings for the Predictive Analytics Module"""

# Data Collection Settings
PROMETHEUS_CONFIG = {
    'host': 'localhost',
    'port': 9090,
    'metrics': {
        'cpu_usage': 'container_cpu_usage_seconds_total',
        'memory_usage': 'container_memory_usage_bytes',
        'network_io': 'container_network_transmit_bytes_total',
        'latency': 'request_duration_seconds'
    },
    'query_timeout': 30,
    'retry_attempts': 3,
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'cpu_usage': (0, 100),
            'memory_usage': (0, 32 * 1024 * 1024 * 1024),
            'network_io': (0, 10 * 1024 * 1024 * 1024),
            'latency': (0, 10)
        }
    }
}

KAFKA_CONFIG = {
    'bootstrap_servers': ['kafka:9092'],
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
            'latency': (0, 10)
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
    'features': ['timestamp', 'cpu_usage', 'memory_usage', 'network_io', 'latency'],
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'cpu_usage': (0, 100),
            'memory_usage': (0, 32 * 1024 * 1024 * 1024),
            'network_io': (0, 10 * 1024 * 1024 * 1024),
            'latency': (0, 1.0)
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
    'features': ['hour', 'day_of_week', 'month', 'cpu_usage', 'memory_usage', 'network_io'],
    'data_validation': {
        'timestamp_format': '%Y-%m-%d %H:%M:%S',
        'value_range': {
            'cpu_usage': (0, 100),
            'memory_usage': (0, 32 * 1024 * 1024 * 1024),
            'network_io': (0, 10 * 1024 * 1024 * 1024),
            'latency': (0, 1.0)
        }
    }
}

# Resource Optimization Settings
OPTIMIZATION_CONFIG = {
    'objective': 'minimize_cost',
    'constraints': {
        'max_cpu': 100,  # percentage
        'max_memory': 32 * 1024 * 1024 * 1024,  # bytes (32GB)
        'max_network': 10 * 1024 * 1024 * 1024,  # bytes (10GB/s)
        'max_latency': 1.0  # seconds
    },
    'weights': {
        'cpu': 0.4,
        'memory': 0.3,
        'network': 0.3
    },
    'solver_config': {
        'solver': 'CBC',
        'safety_margin': 1.2,
        'balance_resources': True,
        'balance_tolerance': 0.2,
        'min_utilization': 20,
        'max_utilization': 90,
        'solution_timeout': 300,
        'options': {
            'msg': 0,
            'timeLimit': 60
        }
    }
}

# Cache Settings
CACHE_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'key_prefix': 'prediction:',
    'expire': 300  # seconds
}

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'predictive_analytics.log',
            'level': 'DEBUG'
        }
    }
} 