#!/usr/bin/env python3
import os
import sys
import time
import joblib
import logging
import argparse
import json
import threading
import psutil
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-runner')

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 初始化Prometheus指标
PREDICTION_COUNT = Counter('model_predictions_total', 'Total prediction requests processed', ['model_id'])
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds', ['model_id'])
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total prediction errors', ['model_id', 'error_type'])
MODEL_LOADED_TIME = Gauge('model_loaded_timestamp', 'Timestamp when model was loaded', ['model_id'])
FEATURE_COUNT = Gauge('model_feature_count', 'Number of features in the model', ['model_id'])
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage of the model service', ['model_id'])
CPU_USAGE = Gauge('model_cpu_usage_percent', 'CPU usage of the model service', ['model_id'])
NETWORK_RX = Counter('model_network_receive_bytes_total', 'Network bytes received', ['model_id'])
NETWORK_TX = Counter('model_network_transmit_bytes_total', 'Network bytes sent', ['model_id'])
NETWORK_RATE = Gauge('model_network_io_bytes_per_second', 'Network IO bytes per second', ['model_id'])

class ModelService:
    def __init__(self, model_id, models_dir="./models"):
        self.model_id = model_id
        self.models_dir = models_dir
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.startup_time = time.time()
        self.load_model()
        
        # 初始化prometheus指标
        PREDICTION_COUNT.labels(model_id=self.model_id)
        PREDICTION_LATENCY.labels(model_id=self.model_id)
        FEATURE_COUNT.labels(model_id=self.model_id).set(len(self.feature_names))
        MODEL_LOADED_TIME.labels(model_id=self.model_id).set(time.time())
        CPU_USAGE.labels(model_id=self.model_id).set(0)
        MEMORY_USAGE.labels(model_id=self.model_id).set(0)
        NETWORK_RX.labels(model_id=self.model_id)
        NETWORK_TX.labels(model_id=self.model_id)
        NETWORK_RATE.labels(model_id=self.model_id).set(0)
        
        # 启动资源监控线程
        self.monitor_thread = None
        self.should_monitor = True
        self.last_net_io = None
        self.start_resource_monitoring()
        
    def load_model(self):
        """加载模型文件"""
        model_path = os.path.join(self.models_dir, f"{self.model_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', [])
                self.metadata = {
                    'upload_time': os.path.getmtime(model_path),
                    'feature_count': len(self.feature_names)
                }
                logger.info(f"Model {self.model_id} loaded successfully with {len(self.feature_names)} features")
                return True
            else:
                logger.error(f"Invalid model format in {model_path}")
                raise ValueError("Invalid model format")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, features):
        """执行模型预测"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # 进行预测
            prediction = self.model.predict([features])[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            PREDICTION_ERRORS.labels(model_id=self.model_id, error_type=type(e).__name__).inc()
            raise

    def start_resource_monitoring(self):
        """启动资源监控线程"""
        def monitor_loop():
            process = psutil.Process(os.getpid())
            while self.should_monitor:
                try:
                    # 监控CPU使用率
                    cpu_percent = process.cpu_percent(interval=0.1)
                    CPU_USAGE.labels(model_id=self.model_id).set(cpu_percent)
                    
                    # 监控内存使用率
                    memory_info = process.memory_info()
                    MEMORY_USAGE.labels(model_id=self.model_id).set(memory_info.rss)
                    
                    # 监控网络I/O
                    try:
                        io_counters = process.io_counters()
                        
                        if self.last_net_io is None:
                            self.last_net_io = {
                                'read_bytes': io_counters.read_bytes,
                                'write_bytes': io_counters.write_bytes,
                                'timestamp': time.time()
                            }
                        else:
                            # 计算增量和速率
                            current_time = time.time()
                            elapsed = current_time - self.last_net_io['timestamp']
                            
                            if elapsed > 0:
                                rx_diff = io_counters.read_bytes - self.last_net_io['read_bytes']
                                tx_diff = io_counters.write_bytes - self.last_net_io['write_bytes']
                                
                                # 增加计数器
                                if rx_diff > 0:
                                    NETWORK_RX.labels(model_id=self.model_id).inc(rx_diff)
                                if tx_diff > 0:
                                    NETWORK_TX.labels(model_id=self.model_id).inc(tx_diff)
                                
                                # 计算速率
                                io_rate = (rx_diff + tx_diff) / elapsed
                                NETWORK_RATE.labels(model_id=self.model_id).set(io_rate)
                                
                                # 更新上次读数
                                self.last_net_io = {
                                    'read_bytes': io_counters.read_bytes,
                                    'write_bytes': io_counters.write_bytes,
                                    'timestamp': current_time
                                }
                    except:
                        # 某些系统可能不支持io_counters
                        pass
                    
                    # 每5秒更新一次指标
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {str(e)}")
                    time.sleep(10)  # 出错后延长间隔
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_resource_monitoring(self):
        """停止资源监控"""
        self.should_monitor = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            logger.info("Resource monitoring stopped")

# 创建模型服务实例
model_service = None

@app.route('/infer', methods=['POST'])
def infer():
    """模型推理接口"""
    global model_service
    
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing features in request"}), 400
        
        features = data["features"]
        
        # 增加指标统计
        PREDICTION_COUNT.labels(model_id=model_service.model_id).inc()
        
        # 执行预测并计时
        start_time = time.time()
        with PREDICTION_LATENCY.labels(model_id=model_service.model_id).time():
            prediction = model_service.predict(features)
        latency = time.time() - start_time
        
        # 更新内存使用指标
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            MEMORY_USAGE.labels(model_id=model_service.model_id).set(memory_info.rss)
        except:
            pass
        
        response = {
            "model_id": model_service.model_id,
            "prediction": prediction,
            "latency": latency,
            "timestamp": time.time()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        PREDICTION_ERRORS.labels(model_id=model_service.model_id, error_type=error_type).inc()
        logger.error(f"Inference error: {error_msg}")
        return jsonify({
            "error": error_msg,
            "error_type": error_type
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    global model_service
    
    status = "healthy" if model_service and model_service.model is not None else "unhealthy"
    uptime = time.time() - model_service.startup_time if model_service else 0
    
    return jsonify({
        "status": status,
        "model_id": model_service.model_id if model_service else None,
        "uptime_seconds": uptime,
        "timestamp": time.time()
    }), 200 if status == "healthy" else 503

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus指标接口"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain'}

@app.route('/metadata', methods=['GET'])
def metadata():
    """模型元数据接口"""
    global model_service
    
    if not model_service:
        return jsonify({"error": "Model service not initialized"}), 500
    
    return jsonify({
        "model_id": model_service.model_id,
        "feature_names": model_service.feature_names,
        "feature_count": len(model_service.feature_names),
        "metadata": model_service.metadata,
        "service_uptime_seconds": time.time() - model_service.startup_time
    }), 200

@app.route('/resources', methods=['GET'])
def resources():
    """资源使用情况接口"""
    global model_service
    
    if not model_service:
        return jsonify({"error": "Model service not initialized"}), 500
    
    try:
        process = psutil.Process(os.getpid())
        
        # 获取最新CPU信息
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # 获取内存信息
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # 获取网络IO信息
        net_io = None
        try:
            io_counters = process.io_counters()
            net_io = {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes
            }
        except:
            # 某些系统可能不支持io_counters
            pass
        
        # 获取启动时间和运行时长
        start_time = process.create_time()
        uptime = time.time() - start_time
        
        return jsonify({
            "model_id": model_service.model_id,
            "resources": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_bytes": memory_info.rss,
                "memory_usage_percent": memory_percent,
                "network_io": net_io,
                "pid": os.getpid(),
                "uptime_seconds": uptime,
                "start_time": start_time
            },
            "timestamp": time.time()
        }), 200
    except Exception as e:
        logger.error(f"Error getting resource usage: {str(e)}")
        return jsonify({
            "error": str(e),
            "model_id": model_service.model_id if model_service else None
        }), 500

def main():
    parser = argparse.ArgumentParser(description='Run a model service for a specific model')
    parser.add_argument('--model_id', required=True, help='ID of the model to serve')
    parser.add_argument('--port', type=int, default=0, help='Port to run the service on (0 for auto)')
    parser.add_argument('--models_dir', default='./models', help='Directory containing model files')
    parser.add_argument('--registry_url', help='URL of the model registry service')
    parser.add_argument('--register', action='store_true', help='Register with registry service')
    
    args = parser.parse_args()
    
    # 初始化模型服务
    global model_service
    try:
        model_service = ModelService(args.model_id, args.models_dir)
        
        # 查找可用端口
        port = args.port
        if port == 0:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()
        
        # 如果指定了注册中心，则向注册中心注册
        if args.register and args.registry_url:
            import requests
            registry_data = {
                "model_id": args.model_id,
                "port": port,
                "status": "running",
                "metadata": model_service.metadata,
                "features": model_service.feature_names
            }
            
            try:
                response = requests.post(
                    f"{args.registry_url}/register", 
                    json=registry_data,
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"Registered model {args.model_id} with registry")
                else:
                    logger.warning(f"Failed to register with registry: {response.text}")
            except Exception as e:
                logger.error(f"Error registering with registry: {str(e)}")
        
        # 创建端口映射文件
        port_file = os.path.join(os.path.dirname(args.models_dir), 'port_mappings', f"{args.model_id}.port")
        os.makedirs(os.path.dirname(port_file), exist_ok=True)
        with open(port_file, 'w') as f:
            f.write(str(port))
        
        logger.info(f"Starting model service for {args.model_id} on port {port}")
        try:
            app.run(host='0.0.0.0', port=port)
        finally:
            # 确保停止资源监控
            if model_service:
                model_service.stop_resource_monitoring()
        
    except Exception as e:
        logger.error(f"Failed to start model service: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 