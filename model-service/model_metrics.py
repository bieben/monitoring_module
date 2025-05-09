#!/usr/bin/env python
"""
模型资源监控模块 - 收集各个模型实例的CPU、内存和网络使用情况
"""

import os
import time
import threading
import logging
import json
import psutil
import requests
from prometheus_client import Gauge, Counter
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-metrics")

# 创建资源使用指标 - 修改名称添加前缀避免冲突
MODEL_CPU_USAGE = Gauge('instance_model_cpu_usage_percent', 'CPU usage percentage', ['model_id', 'instance'])
MODEL_MEMORY_USAGE = Gauge('instance_model_memory_usage_bytes', 'Memory usage in bytes', ['model_id', 'instance'])
MODEL_NETWORK_RX = Counter('instance_model_network_receive_bytes_total', 'Network bytes received', ['model_id', 'instance'])
MODEL_NETWORK_TX = Counter('instance_model_network_transmit_bytes_total', 'Network bytes sent', ['model_id', 'instance'])
MODEL_NETWORK_RATE = Gauge('instance_model_network_io_bytes_per_second', 'Network IO bytes per second', ['model_id', 'instance'])

class ModelMetricsCollector:
    """收集模型服务实例的资源使用指标"""
    
    def __init__(self, refresh_interval: int = 15):
        """
        初始化指标收集器
        
        Args:
            refresh_interval: 刷新间隔(秒)
        """
        self.refresh_interval = refresh_interval
        self.model_processes: Dict[str, Dict[str, Any]] = {}
        self.last_net_io: Dict[str, Dict[str, float]] = {}
        self.running = False
        self.collection_thread = None
        self.registry_url = os.environ.get('REGISTRY_URL', 'http://localhost:5050')
        
    def start(self):
        """启动指标收集"""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Model metrics collector started")
        
    def stop(self):
        """停止指标收集"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Model metrics collector stopped")
            
    def _collection_loop(self):
        """指标收集循环"""
        while self.running:
            try:
                # 同步模型服务进程列表
                self._sync_model_services()
                
                # 收集每个模型的资源使用情况
                self._collect_metrics()
                
                # 等待下一个刷新周期
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠
    
    def _sync_model_services(self):
        """从模型注册中心同步模型服务信息"""
        try:
            # 从注册中心获取运行中的服务
            response = requests.get(f"{self.registry_url}/services", timeout=5)
            if response.status_code == 200:
                services = response.json().get('services', {})
                
                # 更新当前进程信息
                for model_id, service_info in services.items():
                    pid = service_info.get('pid')
                    if pid:
                        # 如果存在PID，尝试查找进程
                        try:
                            process = psutil.Process(pid)
                            self.model_processes[model_id] = {
                                'pid': pid,
                                'process': process,
                                'port': service_info.get('port'),
                                'status': service_info.get('status'),
                                'instance': service_info.get('instance', 'default')
                            }
                        except psutil.NoSuchProcess:
                            # 进程不存在，可能已经停止
                            if model_id in self.model_processes:
                                del self.model_processes[model_id]
                
                # 清理不在服务列表中的进程
                models_to_remove = [model_id for model_id in self.model_processes 
                                   if model_id not in services]
                for model_id in models_to_remove:
                    del self.model_processes[model_id]
                    
                logger.debug(f"Synced {len(self.model_processes)} model services from registry")
        except Exception as e:
            logger.error(f"Failed to sync model services: {str(e)}")
    
    def _collect_metrics(self):
        """收集每个模型的资源使用指标"""
        for model_id, process_info in self.model_processes.items():
            try:
                process = process_info.get('process')
                instance = process_info.get('instance', 'default')
                
                # 使用更通用的方法检查进程是否运行
                # 替换 if not process or not process.is_running():
                if not process:
                    continue
                
                # 尝试获取有效的psutil.Process对象
                psutil_process = None
                try:
                    # 如果是Popen对象或其他类型，尝试获取PID
                    if hasattr(process, 'pid'):
                        pid = process.pid
                        
                        # 尝试从PID创建有效的psutil.Process对象
                        try:
                            psutil_process = psutil.Process(pid)
                            # 确认进程存在
                            psutil_process.status()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                    else:
                        continue
                except (AttributeError, TypeError):
                    continue
                
                # 使用新的psutil_process对象
                process = psutil_process
                
                # 收集CPU使用率
                try:
                    cpu_percent = process.cpu_percent(interval=0.1)
                    MODEL_CPU_USAGE.labels(model_id=model_id, instance=instance).set(cpu_percent)
                except Exception as e:
                    logger.error(f"Error collecting CPU metrics for {model_id}: {str(e)}")
                
                # 收集内存使用情况
                try:
                    memory_info = process.memory_info()
                    MODEL_MEMORY_USAGE.labels(model_id=model_id, instance=instance).set(memory_info.rss)
                except Exception as e:
                    logger.error(f"Error collecting memory metrics for {model_id}: {str(e)}")
                
                # 收集网络I/O
                try:
                    net_io = process.io_counters()
                    
                    # 更新计数器
                    if model_id not in self.last_net_io:
                        self.last_net_io[model_id] = {
                            'read_bytes': net_io.read_bytes,
                            'write_bytes': net_io.write_bytes,
                            'timestamp': time.time()
                        }
                    
                    # 计算速率
                    last_io = self.last_net_io[model_id]
                    elapsed = time.time() - last_io['timestamp']
                    
                    if elapsed > 0:
                        rx_diff = net_io.read_bytes - last_io['read_bytes']
                        tx_diff = net_io.write_bytes - last_io['write_bytes']
                        
                        rx_rate = rx_diff / elapsed
                        tx_rate = tx_diff / elapsed
                        
                        # 更新计数器和速率
                        MODEL_NETWORK_RX.labels(model_id=model_id, instance=instance).inc(rx_diff)
                        MODEL_NETWORK_TX.labels(model_id=model_id, instance=instance).inc(tx_diff)
                        MODEL_NETWORK_RATE.labels(model_id=model_id, instance=instance).set(rx_rate + tx_rate)
                    
                    # 更新上次读数
                    self.last_net_io[model_id] = {
                        'read_bytes': net_io.read_bytes,
                        'write_bytes': net_io.write_bytes,
                        'timestamp': time.time()
                    }
                except Exception as e:
                    logger.error(f"Error collecting network metrics for {model_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics for {model_id}: {str(e)}")

# 单例实例
collector = ModelMetricsCollector()

def start_metrics_collection():
    """启动指标收集"""
    collector.start()

def stop_metrics_collection():
    """停止指标收集"""
    collector.stop()

if __name__ == "__main__":
    # 直接运行时，启动收集器
    start_metrics_collection()
    
    try:
        # 保持程序运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_metrics_collection() 