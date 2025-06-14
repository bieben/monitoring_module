import logging
import os
import joblib
import time
import uuid
import subprocess
import json
import requests
import signal
import psutil
import sys
from typing import Dict, Any, Tuple, List, Optional
from kafka_client import KafkaClient
from flask import jsonify
from prometheus_client import Counter, Histogram, Gauge
from alert_rules import AlertRules
from log_manager import LogManager

class MLService:
    def __init__(self):
        self.model_registry: Dict[str, Dict[str, Any]] = {}  # model_id -> {model, metadata}
        self.log_manager = LogManager(max_cache_size=50000)  # 初始化日志管理器
        self.kafka_client = KafkaClient(log_callback=self.log_manager.add_log)  # 传递日志回调
        self.alert_rules = AlertRules()
        self.start_time = time.time()  # 记录服务启动时间
        
        # 存储模型服务进程
        self.model_processes: Dict[str, Dict[str, Any]] = {}
        
        # 注册中心配置
        self.registry_host = os.environ.get('REGISTRY_HOST', 'localhost')
        self.registry_port = int(os.environ.get('REGISTRY_PORT', '5050'))
        self.registry_url = f"http://{self.registry_host}:{self.registry_port}"
        
        # 端口范围
        self.port_start = int(os.environ.get('MODEL_SERVICE_PORT_START', '8000'))
        self.port_end = int(os.environ.get('MODEL_SERVICE_PORT_END', '8100'))
        self.next_port = self.port_start
        
        # 创建models目录
        os.makedirs("models", exist_ok=True)
        
        # 创建port_mappings目录
        os.makedirs("port_mappings", exist_ok=True)
        
        # 存储活跃告警
        self.active_alerts: List[Dict[str, Any]] = []
        
        # 加载模型注册表
        self._sync_model_registry()
        
        # 加载现有的模型服务进程信息
        self._sync_model_services()

    def _sync_model_registry(self):
        """同步模型注册表"""
        # 清除当前注册表
        self.model_registry.clear()
        
        # 列出models目录中的所有.pkl文件
        try:
            model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
            for model_file in model_files:
                model_id = model_file[:-4]  # 移除.pkl后缀
                try:
                    model_path = os.path.join("models", model_file)
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.model_registry[model_id] = {
                            'model': model_data['model'],
                            'feature_names': model_data.get('feature_names', []),
                            'upload_time': os.path.getmtime(model_path),
                            'prediction_count': 0,
                            'avg_latency': 0.0
                        }
                        logging.info(f"Loaded model {model_id} from {model_path}")
                    else:
                        logging.warning(f"Skipped invalid model file: {model_file}")
                except Exception as e:
                    logging.error(f"Failed to load model {model_file}: {e}")
        except Exception as e:
            logging.error(f"Failed to sync model registry: {e}")

    def _sync_model_services(self):
        """同步已运行的模型服务"""
        try:
            # 查询注册中心获取正在运行的服务列表
            try:
                response = requests.get(f"{self.registry_url}/services", timeout=5)
                if response.status_code == 200:
                    services = response.json().get('services', {})
                    for model_id, service_info in services.items():
                        # 将服务信息添加到进程字典中
                        self.model_processes[model_id] = {
                            'port': service_info.get('port'),
                            'url': service_info.get('url'),
                            'status': service_info.get('status'),
                            'pid': None,  # 运行中的进程PID未知
                            'process': None
                        }
                    logging.info(f"Synced {len(services)} running model services from registry")
            except Exception as e:
                logging.warning(f"Failed to sync with registry: {str(e)}")
                
            # 从端口映射文件加载信息
            port_dir = "port_mappings"
            if os.path.exists(port_dir):
                for filename in os.listdir(port_dir):
                    if filename.endswith('.port'):
                        model_id = filename[:-5]  # 移除.port后缀
                        try:
                            with open(os.path.join(port_dir, filename), 'r') as f:
                                port = int(f.read().strip())
                                if model_id not in self.model_processes:
                                    self.model_processes[model_id] = {
                                        'port': port,
                                        'url': f"http://localhost:{port}",
                                        'status': 'unknown',
                                        'pid': None,
                                        'process': None
                                    }
                        except ValueError:
                            logging.warning(f"Invalid port in file {filename}")
        except Exception as e:
            logging.error(f"Failed to sync model services: {str(e)}")

    def _check_model_file_exists(self, model_id: str) -> bool:
        """检查模型文件是否存在"""
        model_path = os.path.join("models", f"{model_id}.pkl")
        return os.path.exists(model_path)

    def handle_model_upload(self, model_id: str, model_file) -> Tuple[dict, int]:
        """处理模型上传请求"""
        if not model_id or not model_file:
            return {"error": "Missing model_id or file"}, 400

        try:
            save_path = os.path.join("models", f"{model_id}.pkl")
            model_file.save(save_path)

            # 加载模型数据
            model_data = joblib.load(save_path)
            if not isinstance(model_data, dict) or 'model' not in model_data:
                os.remove(save_path)  # 清理无效文件
                return {"error": "Invalid model format"}, 400
            
            # 将模型存储到注册表中
            self.model_registry[model_id] = {
                'model': model_data['model'],
                'feature_names': model_data.get('feature_names', []),
                'upload_time': time.time(),
                'prediction_count': 0,
                'avg_latency': 0.0
            }
            return {"message": f"Model {model_id} uploaded successfully"}, 200
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)  # 清理无效文件
            logging.error(f"Model upload failed: {e}")
            return {"error": str(e)}, 500
        
    def _get_available_port(self) -> int:
        """获取可用端口"""
        import socket
        
        # 首先尝试使用预定义的端口范围
        for port in range(self.next_port, self.port_end + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('', port))
                s.close()
                self.next_port = port + 1
                if self.next_port > self.port_end:
                    self.next_port = self.port_start
                return port
            except:
                continue
        
        # 如果所有预定义端口都在使用，让系统分配
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _start_model_service(self, model_id: str, config: Dict[str, Any]) -> Tuple[bool, str, int]:
        """启动模型服务进程"""
        # 检查模型文件是否存在
        if not self._check_model_file_exists(model_id):
            return False, "Model file not found", 0
        
        # 如果服务已经在运行，返回
        if model_id in self.model_processes and self.model_processes[model_id].get('status') == 'running':
            # 检查服务是否真的在运行
            try:
                port = self.model_processes[model_id]['port']
                url = f"http://localhost:{port}"
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    return True, "Service already running", port
            except:
                # 服务已经不在运行，会继续启动新的服务
                pass
        
        # 获取可用端口
        port = self._get_available_port()
        
        # 构建启动命令
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_runner.py")
        
        # 获取当前工作目录的绝对路径
        cwd = os.getcwd()
        models_dir = os.path.join(cwd, "models")
        
        # 准备环境变量
        env = os.environ.copy()
        
        # 根据配置为服务分配资源
        resources = config.get('resources', {})
        if 'memory_limit' in resources:
            env['MODEL_MEMORY_LIMIT'] = str(resources['memory_limit'])
        if 'cpu_limit' in resources:
            env['MODEL_CPU_LIMIT'] = str(resources['cpu_limit'])
        
        try:
            # 启动服务进程
            cmd = [
                sys.executable,
                script_path,
                "--model_id", model_id,
                "--port", str(port),
                "--models_dir", models_dir,
                "--registry_url", self.registry_url,
                "--register"
            ]
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 存储进程信息
            self.model_processes[model_id] = {
                'port': port,
                'url': f"http://localhost:{port}",
                'status': 'starting',
                'pid': process.pid,
                'process': process,
                'started_at': time.time(),
                'config': config
            }
            
            # 等待服务启动
            max_wait = 30  # 最多等待30秒
            wait_interval = 0.5
            started = False
            
            for _ in range(int(max_wait / wait_interval)):
                time.sleep(wait_interval)
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if response.status_code == 200:
                        started = True
                        self.model_processes[model_id]['status'] = 'running'
                        break
                except:
                    continue
            
            if not started:
                # 服务未能成功启动
                if process.poll() is None:  # 如果进程仍在运行
                    process.terminate()
                return False, "Failed to start service", 0
            
            # 等待服务注册
            time.sleep(1)
            
            return True, "Service started successfully", port
        
        except Exception as e:
            logging.error(f"Failed to start model service: {str(e)}")
            return False, str(e), 0

    def deploy_model(self, model_id: str, config: Dict[str, Any]) -> Tuple[dict, int]:
        """
        部署模型服务
        """
        if not model_id:
            return {"error": "Missing model_id"}, 400

        if model_id not in self.model_registry:
            return {"error": "Model not found. Please upload first."}, 404

        try:
            # 启动模型服务
            success, message, port = self._start_model_service(model_id, config)
            
            if not success:
                return {"error": f"Failed to deploy model: {message}"}, 500
            
            # 获取模型服务URL
            service_url = f"http://localhost:{port}"
            
            # 更新模型信息
            model_info = self.model_registry[model_id]
            model_info['deployed'] = True
            model_info['deploy_config'] = config
            model_info['deploy_time'] = time.time()
            model_info['service_url'] = service_url
            model_info['service_port'] = port

            return {
                "message": f"Model {model_id} deployed successfully",
                "service_url": service_url,
                "port": port,
                "config": config
            }, 200
        except Exception as e:
            logging.error(f"Deployment failed for model {model_id}: {e}")
            return {"error": str(e)}, 500

    def _stop_model_service(self, model_id: str) -> Tuple[bool, str]:
        """停止模型服务"""
        if model_id not in self.model_processes:
            return False, "Service not found"
        
        service_info = self.model_processes[model_id]
        pid = service_info.get('pid')
        process = service_info.get('process')
        port = service_info.get('port')
        
        success = False
        message = ""
        
        try:
            # 先尝试通过注册中心注销
            try:
                response = requests.post(f"{self.registry_url}/unregister/{model_id}", timeout=5)
                if response.status_code == 200:
                    logging.info(f"Unregistered model service {model_id} from registry")
            except Exception as e:
                logging.warning(f"Failed to unregister from registry: {str(e)}")
            
            # 如果有进程对象，先尝试优雅终止
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)  # 等待进程终止
                    success = True
                except subprocess.TimeoutExpired:
                    # 如果进程没有及时终止，强制杀死
                    process.kill()
                    success = True
            
            # 如果有PID但没有进程对象，尝试使用psutil终止
            elif pid:
                try:
                    if psutil.pid_exists(pid):
                        p = psutil.Process(pid)
                        p.terminate()
                        gone, alive = psutil.wait_procs([p], timeout=3)
                        if alive:
                            for p in alive:
                                p.kill()
                        success = True
                except:
                    pass
            
            # 尝试通过端口查找进程并终止
            if not success and port:
                try:
                    for proc in psutil.process_iter(['pid', 'connections']):
                        try:
                            for conn in proc.connections():
                                if conn.laddr.port == port:
                                    proc.terminate()
                                    gone, alive = psutil.wait_procs([proc], timeout=3)
                                    if alive:
                                        for p in alive:
                                            p.kill()
                                    success = True
                                    break
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            continue
                except:
                    pass
            
            # 清除进程信息
            self.model_processes.pop(model_id, None)
            
            # 删除端口映射文件
            port_file = os.path.join("port_mappings", f"{model_id}.port")
            if os.path.exists(port_file):
                os.remove(port_file)
            
            message = "Service stopped successfully"
            return success, message
        
        except Exception as e:
            message = f"Error stopping service: {str(e)}"
            logging.error(message)
            return False, message

    def handle_prediction(self, data: dict) -> Tuple[dict, int]:
        """处理预测请求"""
        if not data or "model_id" not in data:
            return {"error": "Missing model_id"}, 400

        model_id = data["model_id"]
        
        # 检查模型服务是否已部署
        if model_id in self.model_processes and self.model_processes[model_id].get('status') == 'running':
            # 使用模型服务进行预测
            try:
                service_url = self.model_processes[model_id]['url']
                response = requests.post(
                    f"{service_url}/infer",
                    json={"features": data.get("features", [])},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # 从独立服务预测成功，记录日志
                    response_data = response.json()
                    try:
                        # 提取响应中的预测值和延迟
                        prediction = response_data.get("prediction", 0.0)
                        latency = response_data.get("latency", 0.0)
                        
                        # 记录日志
                        log_data = {
                            "model_id": model_id,
                            "prediction": prediction,
                            "latency": latency,
                            "features": data.get("features", []),
                            "timestamp": time.time(),
                            "level": "INFO" if latency < 1.0 else "WARNING",
                            "message": f"Service prediction for model {model_id} completed in {latency:.3f}s"
                        }
                        # 直接添加到日志管理器
                        self.log_manager.add_log(log_data)
                        
                        # 尝试通过Kafka发送
                        kafka_enabled = os.environ.get('KAFKA_ENABLED', 'false').lower() == 'true'
                        if kafka_enabled:
                            self.kafka_client.send_log(log_data)
                    except Exception as log_err:
                        logging.warning(f"Failed to log service prediction: {str(log_err)}")
                        
                    return response_data, 200
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    
                    # 记录错误日志
                    log_data = {
                        "model_id": model_id,
                        "error": error_msg,
                        "features": data.get("features", []),
                        "timestamp": time.time(),
                        "level": "ERROR",
                        "message": f"Service prediction error for model {model_id}: {error_msg}"
                    }
                    # 直接添加到日志管理器
                    self.log_manager.add_log(log_data)
                    
                    return {"error": f"Prediction service error: {error_msg}"}, response.status_code
            
            except Exception as e:
                logging.error(f"Error forwarding prediction to service: {str(e)}")
                
                # 记录服务连接错误日志
                log_data = {
                    "model_id": model_id,
                    "error": str(e),
                    "features": data.get("features", []),
                    "timestamp": time.time(),
                    "level": "ERROR",
                    "message": f"Service connection error for model {model_id}: {str(e)}"
                }
                # 直接添加到日志管理器
                self.log_manager.add_log(log_data)
                
                return {"error": f"Failed to reach prediction service: {str(e)}"}, 500
        
        # 如果服务未部署或不可用，尝试使用本地模型
        if not self._check_model_file_exists(model_id):
            if model_id in self.model_registry:
                del self.model_registry[model_id]  # 如果文件丢失，从注册表中删除
                
            # 记录模型文件缺失错误
            log_data = {
                "model_id": model_id,
                "error": "Model file not found",
                "features": data.get("features", []),
                "timestamp": time.time(),
                "level": "ERROR",
                "message": f"Model file not found for {model_id}"
            }
            # 直接添加到日志管理器
            self.log_manager.add_log(log_data)
            
            return {"error": "Model file not found"}, 404

        start_time = time.time()
        try:
            # 获取模型和元数据
            model_info = self.model_registry.get(model_id)
            if not model_info:
                # 记录模型未注册错误
                log_data = {
                    "model_id": model_id,
                    "error": "Model not registered",
                    "features": data.get("features", []),
                    "timestamp": time.time(),
                    "level": "ERROR",
                    "message": f"Model {model_id} not registered"
                }
                # 直接添加到日志管理器
                self.log_manager.add_log(log_data)
                
                return {"error": "Model not registered"}, 404

            model = model_info['model']
            prediction = float(model.predict([data["features"]])[0])
            latency = time.time() - start_time

            # 更新模型统计信息
            model_info['prediction_count'] += 1
            model_info['avg_latency'] = (
                (model_info['avg_latency'] * (model_info['prediction_count'] - 1) + latency)
                / model_info['prediction_count']
            )

            result = {
                "prediction": prediction,
                "latency": latency,
                "model_stats": {
                    "total_predictions": model_info['prediction_count'],
                    "avg_latency": model_info['avg_latency']
                },
                "note": "Using local model (service not deployed)"
            }

            # 记录预测日志
            try:
                # 直接添加到日志管理器
                log_data = {
                    "model_id": model_id,
                    "prediction": prediction,
                    "latency": latency,
                    "features": data.get("features", []),
                    "timestamp": time.time(),
                    "level": "INFO" if latency < 1.0 else "WARNING",
                    "message": f"Local prediction for model {model_id} completed in {latency:.3f}s"
                }
                self.log_manager.add_log(log_data)
                
                # 尝试通过Kafka发送
                kafka_enabled = os.environ.get('KAFKA_ENABLED', 'false').lower() == 'true'
                if kafka_enabled:
                    self.kafka_client.send_log(log_data)
                    
            except Exception as log_err:
                logging.warning(f"Failed to log prediction: {str(log_err)}")

            return result, 200
        except Exception as e:
            logging.error(f"Prediction failed for model {model_id}: {e}")
            return {"error": str(e)}, 500

    def delete_model(self, model_id: str) -> Tuple[dict, int]:
        """从文件系统和注册表中删除模型"""
        if not model_id:
            return {"error": "Missing model_id"}, 400

        try:
            # 如果模型服务正在运行，先停止
            if model_id in self.model_processes:
                success, message = self._stop_model_service(model_id)
                if not success:
                    logging.warning(f"Could not stop model service: {message}")
            
            # 删除模型文件
            model_path = os.path.join("models", f"{model_id}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
                if model_id in self.model_registry:
                    del self.model_registry[model_id]
                return {"message": f"Model {model_id} deleted successfully"}, 200
            else:
                if model_id in self.model_registry:
                    del self.model_registry[model_id]
                return {"error": "Model file not found"}, 404
        except Exception as e:
            logging.error(f"Failed to delete model {model_id}: {e}")
            return {"error": str(e)}, 500

    def stop_deployment(self, model_id: str) -> Tuple[dict, int]:
        """停止模型部署，但不删除模型文件"""
        if not model_id:
            return {"error": "Missing model_id"}, 400

        try:
            # 更新模型信息
            if model_id in self.model_registry:
                model_info = self.model_registry[model_id]
                model_info['deployed'] = False
                if 'deploy_config' in model_info:
                    del model_info['deploy_config']
                if 'deploy_time' in model_info:
                    del model_info['deploy_time']
                if 'service_url' in model_info:
                    del model_info['service_url']
                if 'service_port' in model_info:
                    del model_info['service_port']
            
            # 停止模型服务
            if model_id in self.model_processes:
                success, message = self._stop_model_service(model_id)
                if not success:
                    return {"error": f"Failed to stop model service: {message}"}, 500
            
            return {"message": f"Model {model_id} deployment stopped successfully"}, 200
        except Exception as e:
            logging.error(f"Failed to stop deployment for model {model_id}: {e}")
            return {"error": str(e)}, 500

    def get_models_status(self) -> Tuple[dict, int]:
        """获取所有已注册模型的状态"""
        try:
            status = {}
            for model_id, info in self.model_registry.items():
                # 检查模型是否已部署
                deployed = False
                service_info = {}
                
                if model_id in self.model_processes:
                    process_info = self.model_processes[model_id]
                    deployed = process_info.get('status') == 'running'
                    
                    # 检查服务是否真的在运行
                    try:
                        port = process_info.get('port')
                        if port:
                            url = f"http://localhost:{port}"
                            response = requests.get(f"{url}/health", timeout=2)
                            if response.status_code == 200:
                                service_info = {
                                    'url': url,
                                    'port': port,
                                    'health': response.json(),
                                    'uptime': time.time() - process_info.get('started_at', time.time())
                                }
                            else:
                                deployed = False
                    except:
                        deployed = False
                
                status[model_id] = {
                    'status': 'active' if deployed else 'inactive',
                    'metadata': {
                        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                   time.localtime(info['upload_time'])),
                        'feature_names': info['feature_names'],
                        'feature_count': len(info['feature_names'])
                    },
                    'performance': {
                        'total_predictions': info['prediction_count'],
                        'avg_latency_ms': round(info['avg_latency'] * 1000, 2),  # 转换为毫秒
                        'last_prediction': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                       time.localtime(time.time())) if info['prediction_count'] > 0 else 'Never'
                    },
                    'deployment': {
                        'deployed': deployed,
                        'deploy_config': info.get('deploy_config', {}),
                        'deploy_time': time.strftime('%Y-%m-%d %H:%M:%S',
                                                    time.localtime(info['deploy_time']))
                        if 'deploy_time' in info else 'Not deployed',
                        'service': service_info if deployed else {}
                    }
                }
            
            # 从注册表获取可能不在本地注册的服务
            try:
                response = requests.get(f"{self.registry_url}/services", timeout=3)
                if response.status_code == 200:
                    registry_services = response.json().get('services', {})
                    for model_id, service_info in registry_services.items():
                        if model_id not in status:
                            # 这个服务不在本地注册表中，添加一个简单的状态
                            status[model_id] = {
                                'status': service_info.get('status', 'unknown'),
                                'metadata': {
                                    'upload_time': 'Unknown',
                                    'feature_names': [],
                                    'feature_count': 0
                                },
                                'performance': {
                                    'total_predictions': 0,
                                    'avg_latency_ms': 0,
                                    'last_prediction': 'Never'
                                },
                                'deployment': {
                                    'deployed': True,
                                    'deploy_config': {},
                                    'deploy_time': 'Unknown',
                                    'service': {
                                        'url': service_info.get('url'),
                                        'port': service_info.get('port'),
                                        'registered_at': service_info.get('registered_at')
                                    }
                                }
                            }
            except Exception as e:
                logging.warning(f"Failed to get services from registry: {str(e)}")
            
            return {
                "models": status,
                "total_models": len(status)
            }, 200
        except Exception as e:
            logging.error(f"Failed to get models status: {e}")
            return {"error": str(e)}, 500
            
    def _log_prediction(self, model_id: str, prediction: float, latency: float, features: list):
        """记录预测信息"""
        try:
            # 检查Kafka是否启用
            kafka_enabled = os.environ.get('KAFKA_ENABLED', 'false').lower() == 'true'
            if not kafka_enabled:
                logging.debug(f"Kafka logging disabled by environment variable")
                return
                
            # 使用Kafka记录预测结果
            log_data = {
                "model_id": model_id,
                "prediction": prediction,
                "latency": latency,
                "features": features,
                "timestamp": time.time(),
                "level": "INFO" if latency < 1.0 else "WARNING",
                "message": f"Prediction for model {model_id} completed in {latency:.3f}s"
            }
            self.kafka_client.send_log(log_data)
            
            # 直接添加到日志管理器
            self.log_manager.add_log(log_data)
        except Exception as e:
            logging.error(f"Failed to log prediction: {e}")
            # 错误不影响主业务逻辑，所以只记录不抛出 

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取当前活跃的告警列表，使用真实的系统资源监控数据
        
        返回:
            List[Dict[str, Any]]: 告警列表，每个告警包含以下字段：
                - name: 告警名称
                - level: 告警级别 (CRITICAL, WARNING, INFO)
                - message: 告警消息
                - type: 告警类型
                - value: 当前值
                - threshold: 阈值
                - active: 是否活跃
                - model_id: 相关模型ID
                - timestamp: 告警时间戳
        """
        # 清除所有现有告警，确保使用最新的阈值重新评估
        self.active_alerts = []
        
        # 获取系统整体资源使用情况 - 始终获取系统整体资源数据
        system_cpu_usage = psutil.cpu_percent(interval=0.1)
        system_memory_usage = psutil.virtual_memory().percent
        system_disk_usage = psutil.disk_usage('/').percent
        
        # 检查所有已注册模型的状态，生成系统检查告警
        for model_id, model_info in self.model_registry.items():
            # 检查模型资源使用情况
            try:
                # 标记是否已检查模型特定的资源使用情况
                checked_model_resources = False
                
                # 获取真实资源使用情况
                if model_id in self.model_processes:
                    process_info = self.model_processes[model_id]
                    if process_info.get('status') == 'running' and 'process' in process_info:
                        process = process_info['process']
                        
                        # 确保进程对象可用，并获取其PID
                        if hasattr(process, 'pid'):
                            try:
                                # 使用psutil获取进程的真实CPU和内存使用率
                                psutil_process = psutil.Process(process.pid)
                                
                                # 获取CPU使用率
                                cpu_usage = psutil_process.cpu_percent(interval=0.1)
                                
                                # 获取内存使用率
                                memory_usage = psutil_process.memory_percent()
                                
                                # 检查资源使用情况并生成告警
                                resource_alerts = self.alert_rules.check_resource_usage(model_id, cpu_usage, memory_usage)
                                
                                # 将生成的资源告警添加到活跃告警列表
                                for alert in resource_alerts:
                                    self.active_alerts.append({
                                        "name": f"{alert['type'].replace('_', ' ').title()}",
                                        "level": alert['level'],
                                        "message": alert['message'],
                                        "type": alert['type'],
                                        "value": cpu_usage if 'cpu' in alert['type'] else memory_usage,
                                        "threshold": self.alert_rules.config.get_config('resource_rules')['cpu_threshold' if 'cpu' in alert['type'] else 'memory_threshold'],
                                        "active": True,
                                        "model_id": model_id,
                                        "timestamp": time.time(),
                                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                    })
                                
                                checked_model_resources = True
                                
                                # 即使检查了模型资源，也添加对系统整体资源的检查
                                # 这确保即使模型资源使用率为0也能检测到系统级别的高资源使用率
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                                logging.warning(f"无法获取进程 {process.pid} 的资源使用情况: {str(e)}")
                
                # 使用系统整体资源数据进行检查 - 现在对每个模型都执行此检查，无论特定模型进程是否可用
                # 系统CPU使用率检查
                resource_rules = self.alert_rules.config.get_config('resource_rules')
                cpu_threshold = resource_rules.get('cpu_threshold', 80.0)
                memory_threshold = resource_rules.get('memory_threshold', 85.0)
                disk_threshold = resource_rules.get('disk_threshold', 90.0)
                
                # 检查CPU使用率
                if system_cpu_usage > cpu_threshold:
                    self.active_alerts.append({
                        "name": "System CPU Usage",
                        "level": "WARNING",
                        "message": f"System CPU usage ({system_cpu_usage:.1f}%) exceeds threshold ({cpu_threshold}%)",
                        "type": "system_cpu_usage",
                        "value": system_cpu_usage,
                        "threshold": cpu_threshold,
                        "active": True,
                        "model_id": model_id,
                        "timestamp": time.time(),
                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                
                # 检查内存使用率
                if system_memory_usage > memory_threshold:
                    self.active_alerts.append({
                        "name": "System Memory Usage",
                        "level": "WARNING",
                        "message": f"System memory usage ({system_memory_usage:.1f}%) exceeds threshold ({memory_threshold}%)",
                        "type": "system_memory_usage",
                        "value": system_memory_usage,
                        "threshold": memory_threshold,
                        "active": True,
                        "model_id": model_id,
                        "timestamp": time.time(),
                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                
                # 检查磁盘使用率
                if system_disk_usage > disk_threshold:
                    self.active_alerts.append({
                        "name": "System Disk Usage",
                        "level": "WARNING",
                        "message": f"System disk usage ({system_disk_usage:.1f}%) exceeds threshold ({disk_threshold}%)",
                        "type": "system_disk_usage",
                        "value": system_disk_usage,
                        "threshold": disk_threshold,
                        "active": True,
                        "model_id": model_id,
                        "timestamp": time.time(),
                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                
            except Exception as e:
                logging.error(f"获取模型 {model_id} 资源使用情况时出错: {str(e)}", exc_info=True)
        
        # 移除过期的告警（例如，超过30分钟的告警）
        current_time = time.time()
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if current_time - alert.get('timestamp', 0) < 1800  # 30分钟
        ]
        
        return self.active_alerts 