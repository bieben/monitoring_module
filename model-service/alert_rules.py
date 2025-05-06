import logging
from typing import Dict, Any, List, Optional
from prometheus_client import Counter, Histogram, Gauge
from monitoring_config import MonitoringConfig

class AlertRules:
    def __init__(self):
        # 初始化配置管理器
        self.config = MonitoringConfig()
        
        # 初始化 Prometheus 指标
        self.error_counter = Counter('model_errors_total', 'Total error count', ['model_id', 'error_type'])
        self.memory_usage = Gauge('model_memory_usage_bytes', 'Memory usage in bytes', ['model_id'])
        self.cpu_usage = Gauge('model_cpu_usage_percent', 'CPU usage percentage', ['model_id'])
        self.prediction_accuracy = Gauge('model_prediction_accuracy', 'Model prediction accuracy', ['model_id'])
        
        # 初始化监控状态
        self.monitoring_state = {
            'error_counts': {},            # 错误计数
            'consecutive_errors': {},       # 连续错误计数
            'last_accuracy': {},           # 上次准确率
            'health_status': {}            # 健康状态
        }

    def check_latency(self, model_id: str, latency: float) -> List[Dict[str, Any]]:
        """检查延迟并返回告警列表"""
        alerts = []
        latency_rules = self.config.get_config('latency_rules')
        
        if latency > latency_rules['critical']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Model {model_id} latency ({latency}s) exceeds critical threshold ({latency_rules["critical"]}s)',
                'type': 'latency'
            })
        elif latency > latency_rules['warning']:
            alerts.append({
                'level': 'WARNING',
                'message': f'Model {model_id} latency ({latency}s) exceeds warning threshold ({latency_rules["warning"]}s)',
                'type': 'latency'
            })
        
        return alerts

    def check_errors(self, model_id: str, error_type: str = 'general') -> List[Dict[str, Any]]:
        """检查错误率并返回告警列表"""
        alerts = []
        error_rules = self.config.get_config('error_rules')
        
        # 更新错误计数
        if model_id not in self.monitoring_state['error_counts']:
            self.monitoring_state['error_counts'][model_id] = []
        
        self.monitoring_state['error_counts'][model_id].append(1)
        if len(self.monitoring_state['error_counts'][model_id]) > error_rules['error_window']:
            self.monitoring_state['error_counts'][model_id].pop(0)
        
        # 计算错误率
        error_rate = sum(self.monitoring_state['error_counts'][model_id]) / len(self.monitoring_state['error_counts'][model_id])
        
        if error_rate > error_rules['max_error_rate']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Model {model_id} error rate ({error_rate:.2%}) exceeds threshold ({error_rules["max_error_rate"]:.2%})',
                'type': 'error_rate'
            })
        
        # 更新 Prometheus 指标
        self.error_counter.labels(model_id=model_id, error_type=error_type).inc()
        
        return alerts

    def check_resource_usage(self, model_id: str, cpu_usage: float, memory_usage: float) -> List[Dict[str, Any]]:
        """检查资源使用情况并返回告警列表"""
        alerts = []
        resource_rules = self.config.get_config('resource_rules')
        
        if cpu_usage > resource_rules['cpu_threshold']:
            alerts.append({
                'level': 'WARNING',
                'message': f'Model {model_id} CPU usage ({cpu_usage:.1f}%) exceeds threshold ({resource_rules["cpu_threshold"]}%)',
                'type': 'cpu_usage'
            })
        
        if memory_usage > resource_rules['memory_threshold']:
            alerts.append({
                'level': 'WARNING',
                'message': f'Model {model_id} memory usage ({memory_usage:.1f}%) exceeds threshold ({resource_rules["memory_threshold"]}%)',
                'type': 'memory_usage'
            })
        
        # 更新 Prometheus 指标
        self.cpu_usage.labels(model_id=model_id).set(cpu_usage)
        self.memory_usage.labels(model_id=model_id).set(memory_usage)
        
        return alerts

    def check_model_performance(self, model_id: str, accuracy: float, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查模型性能并返回告警列表"""
        alerts = []
        performance_rules = self.config.get_config('model_performance_rules')
        
        # 检查准确率
        if accuracy < performance_rules['min_accuracy']:
            alerts.append({
                'level': 'WARNING',
                'message': f'Model {model_id} accuracy ({accuracy:.2%}) below minimum threshold ({performance_rules["min_accuracy"]:.2%})',
                'type': 'accuracy'
            })
        
        # 检查准确率下降
        if model_id in self.monitoring_state['last_accuracy']:
            accuracy_drop = self.monitoring_state['last_accuracy'][model_id] - accuracy
            if accuracy_drop > performance_rules['accuracy_drop_threshold']:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'Model {model_id} accuracy dropped by {accuracy_drop:.2%}',
                    'type': 'accuracy_drop'
                })
        
        self.monitoring_state['last_accuracy'][model_id] = accuracy
        
        # 更新 Prometheus 指标
        self.prediction_accuracy.labels(model_id=model_id).set(accuracy)
        
        return alerts

    def check_system_health(self, model_id: str, dependencies: List[str]) -> List[Dict[str, Any]]:
        """检查系统健康状态并返回告警列表"""
        alerts = []
        health_rules = self.config.get_config('health_rules')
        
        # 检查依赖服务
        for dep in dependencies:
            try:
                # 这里应该实现实际的依赖服务检查逻辑
                # 现在只是一个占位符
                is_healthy = True  # check_dependency(dep)
                
                if not is_healthy:
                    alerts.append({
                        'level': 'CRITICAL',
                        'message': f'Model {model_id} dependency {dep} is unhealthy',
                        'type': 'dependency'
                    })
            except Exception as e:
                alerts.append({
                    'level': 'CRITICAL',
                    'message': f'Model {model_id} dependency {dep} check failed: {str(e)}',
                    'type': 'dependency'
                })
        
        return alerts

    def process_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """处理告警信息"""
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                logging.critical(alert['message'])
            elif alert['level'] == 'WARNING':
                logging.warning(alert['message'])
            else:
                logging.info(alert['message'])

    def update_config(self, category: str, key: str, value: Any) -> bool:
        """更新配置"""
        return self.config.update_config(category, key, value)

    def update_category(self, category: str, values: Dict[str, Any]) -> bool:
        """更新类别配置"""
        return self.config.update_category(category, values)

    def get_config(self, category: Optional[str] = None, key: Optional[str] = None) -> Any:
        """获取配置"""
        return self.config.get_config(category, key)

    def reset_config(self, category: Optional[str] = None) -> bool:
        """重置配置"""
        return self.config.reset_to_default(category) 