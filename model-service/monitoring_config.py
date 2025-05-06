from typing import Dict, Any, Optional
import json
import os
import logging
import copy

class MonitoringConfig:
    def __init__(self, config_file: str = "monitoring_config.json"):
        self.config_file = config_file
        self.default_config = {
            "latency_rules": {
                "critical": 5.0,
                "warning": 1.0,
                "batch": 30.0
            },
            "error_rules": {
                "max_error_rate": 0.05,
                "consecutive_errors": 5,
                "error_window": 100
            },
            "resource_rules": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "disk_threshold": 90.0
            },
            "model_performance_rules": {
                "min_accuracy": 0.90,
                "accuracy_drop_threshold": 0.05,
                "feature_drift_threshold": 0.1
            },
            "health_rules": {
                "max_startup_time": 300,
                "healthcheck_interval": 60,
                "dependency_timeout": 10
            }
        }
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """从文件加载配置，如果文件不存在则使用默认配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # 确保所有必要的配置项都存在
                for category, values in self.default_config.items():
                    if category not in config:
                        config[category] = values
                    else:
                        for key, value in values.items():
                            if key not in config[category]:
                                config[category][key] = value
                return config
            else:
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.default_config

    def save_config(self, config: Dict[str, Any]) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            return False

    def update_config(self, category: str, key: str, value: Any) -> bool:
        """更新单个配置项"""
        try:
            if category not in self.config:
                return False
            if key not in self.config[category]:
                return False
            
            # 验证值的类型
            original_type = type(self.config[category][key])
            if not isinstance(value, original_type):
                try:
                    value = original_type(value)
                except:
                    return False
            
            self.config[category][key] = value
            return self.save_config(self.config)
        except Exception as e:
            logging.error(f"Error updating config: {e}")
            return False

    def update_category(self, category: str, values: Dict[str, Any]) -> bool:
        """更新整个类别的配置"""
        try:
            if category not in self.config:
                return False
            
            # 验证所有的键都是有效的
            for key in values:
                if key not in self.config[category]:
                    return False
            
            # 更新配置
            self.config[category].update(values)
            return self.save_config(self.config)
        except Exception as e:
            logging.error(f"Error updating category: {e}")
            return False

    def get_config(self, category: Optional[str] = None, key: Optional[str] = None) -> Any:
        """获取配置值"""
        try:
            if category is None:
                return self.config
            if key is None:
                return self.config.get(category)
            return self.config.get(category, {}).get(key)
        except Exception as e:
            logging.error(f"Error getting config: {e}")
            return None

    def reset_to_default(self, category: Optional[str] = None) -> bool:
        """重置配置到默认值"""
        try:
            if category is None:
                logging.info("Resetting all configuration to default values")
                self.config = copy.deepcopy(self.default_config)
            elif category in self.config:
                logging.info(f"Resetting category '{category}' to default values")
                self.config[category] = copy.deepcopy(self.default_config[category])
            else:
                logging.error(f"Invalid category '{category}' for reset")
                return False
            
            # 保存前记录配置变更
            logging.info(f"Saving reset configuration to {self.config_file}")
            if self.save_config(self.config):
                logging.info("Configuration reset successful")
                return True
            else:
                logging.error("Failed to save reset configuration")
                return False
        except Exception as e:
            logging.error(f"Error resetting config: {str(e)}")
            return False 