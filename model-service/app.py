from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import logging
import threading
import argparse
import os
import sys
import time
from service import MLService
from flask_cors import CORS
from model_registry import run_registry_service
from model_metrics import start_metrics_collection, stop_metrics_collection
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model-service')

# 创建应用
app = Flask(__name__)
CORS(app)  # 启用 CORS

# Prometheus 指标
REQUEST_COUNT = Counter('model_service_requests_total', 'Total model service requests', ['endpoint', 'model_id'])
RESPONSE_TIME = Histogram('model_service_response_time_seconds', 'Response time in seconds', ['endpoint', 'model_id'])
ACTIVE_SERVICES = Gauge('model_service_active_count', 'Number of active model services')
# 注意：model_errors_total指标已在alert_rules.py中定义

# 初始化服务
ml_service = MLService()
# 使用MLService中的alert_rules访问错误计数器
model_errors_counter = ml_service.alert_rules.error_counter

# 更新活跃服务数量
def update_active_services_metric():
    count = sum(1 for _, process_info in ml_service.model_processes.items() 
               if process_info.get('status') == 'running')
    ACTIVE_SERVICES.set(count)

# 初始化请求计数器
def init_metrics():
    for model_id in ml_service.model_registry.keys():
        REQUEST_COUNT.labels(endpoint="/predict", model_id=model_id)
        RESPONSE_TIME.labels(endpoint="/predict", model_id=model_id)
    
    REQUEST_COUNT.labels(endpoint="/upload_model", model_id="global")
    RESPONSE_TIME.labels(endpoint="/upload_model", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/deploy", model_id="global")
    RESPONSE_TIME.labels(endpoint="/deploy", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/delete_model", model_id="global")
    RESPONSE_TIME.labels(endpoint="/delete_model", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/stop_deployment", model_id="global")
    RESPONSE_TIME.labels(endpoint="/stop_deployment", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/models/status", model_id="global")
    RESPONSE_TIME.labels(endpoint="/models/status", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/health", model_id="global")
    RESPONSE_TIME.labels(endpoint="/health", model_id="global")
    
    REQUEST_COUNT.labels(endpoint="/metrics", model_id="global")
    # 注意：不要为 /metrics 添加响应时间监控，这会导致递归问题

init_metrics()

@app.route("/upload_model", methods=["POST"])
def upload_model():
    """上传模型文件"""
    model_id = request.form.get("model_id")
    
    with RESPONSE_TIME.labels(endpoint="/upload_model", model_id="global").time():
        REQUEST_COUNT.labels(endpoint="/upload_model", model_id="global").inc()
        result, status_code = ml_service.handle_model_upload(model_id, request.files.get("file"))
    
    # 如果上传成功，初始化该模型的指标
    if status_code == 200 and model_id:
        REQUEST_COUNT.labels(endpoint="/predict", model_id=model_id)
        RESPONSE_TIME.labels(endpoint="/predict", model_id=model_id)
    
    return jsonify(result), status_code

@app.route("/deploy", methods=["POST"])
def deploy():
    """部署模型为独立服务"""
    with RESPONSE_TIME.labels(endpoint="/deploy", model_id="global").time():
        REQUEST_COUNT.labels(endpoint="/deploy", model_id="global").inc()
        
        data = request.get_json()
        if not data or not data.get("model_id"):
            return jsonify({"error": "Missing model_id"}), 400
            
        model_id = data.get("model_id")
        environment = data.get("environment", "production")
        
        # 默认资源配置
        resources = data.get("resources", {
            "cpu_limit": "2",
            "memory_limit": "512MB",
            "timeout": 60
        })
        
        config = {
            "environment": environment,
            "resources": resources,
            "created_at": time.time(),
            "created_by": request.headers.get("X-User", "unknown")
        }
        
        result, status_code = ml_service.deploy_model(model_id, config)
        
        # 更新活跃服务数量指标
        update_active_services_metric()
        
        return jsonify(result), status_code

@app.route("/predict", methods=["POST"])
def predict():
    """处理预测请求（可选择转发到独立服务）"""
    data = request.get_json()
    if not data or "model_id" not in data:
        return jsonify({"error": "Missing model_id"}), 400
    
    model_id = data["model_id"]
    
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/predict", model_id=model_id).inc()
    
    # 执行预测并计时
    with RESPONSE_TIME.labels(endpoint="/predict", model_id=model_id).time():
        try:
            result, status_code = ml_service.handle_prediction(data)
            
            # 处理非异常错误 (404, 500等)
            if status_code >= 400:
                error_type = "not_found" if status_code == 404 else "prediction"
                model_errors_counter.labels(model_id=model_id, error_type=error_type).inc()
                logger.error(f"Prediction error for model {model_id}: {result.get('error', 'Unknown error')}")
                
            return jsonify(result), status_code
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id=model_id, error_type="prediction").inc()
            # 记录详细错误日志
            logger.error(f"Prediction error for model {model_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/models/status")
def models_status():
    """获取所有已注册模型的状态"""
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/models/status", model_id="global").inc()
    
    # 执行状态获取并计时
    with RESPONSE_TIME.labels(endpoint="/models/status", model_id="global").time():
        try:
            # 更新活跃服务数量指标
            update_active_services_metric()
            
            result, status_code = ml_service.get_models_status()
            return jsonify(result), status_code
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id="global", error_type="status").inc()
            # 记录详细错误日志
            logger.error(f"Failed to get models status: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to get models status: {str(e)}"}), 500

@app.route("/delete_model/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """删除模型"""
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/delete_model", model_id="global").inc()
    
    # 执行删除操作并计时
    with RESPONSE_TIME.labels(endpoint="/delete_model", model_id="global").time():
        try:
            result, status_code = ml_service.delete_model(model_id)
            
            # 处理非异常错误
            if status_code >= 400:
                error_type = "not_found" if status_code == 404 else "deletion"
                model_errors_counter.labels(model_id=model_id, error_type=error_type).inc()
                logger.error(f"Model deletion error for {model_id}: {result.get('error', 'Unknown error')}")
            
            # 更新活跃服务数量指标
            update_active_services_metric()
            
            return jsonify(result), status_code
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id=model_id, error_type="deletion").inc()
            # 记录详细错误日志
            logger.error(f"Model deletion error for {model_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Model deletion failed: {str(e)}"}), 500

@app.route("/stop_deployment/<model_id>", methods=["POST"])
def stop_deployment(model_id):
    """停止模型部署但不删除模型"""
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/stop_deployment", model_id="global").inc()
    
    # 执行停止操作并计时
    with RESPONSE_TIME.labels(endpoint="/stop_deployment", model_id="global").time():
        try:
            result, status_code = ml_service.stop_deployment(model_id)
            
            # 处理非异常错误
            if status_code >= 400:
                error_type = "not_found" if status_code == 404 else "stop_deployment"
                model_errors_counter.labels(model_id=model_id, error_type=error_type).inc()
                logger.error(f"Stop deployment error for {model_id}: {result.get('error', 'Unknown error')}")
            
            # 更新活跃服务数量指标
            update_active_services_metric()
            
            return jsonify(result), status_code
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id=model_id, error_type="stop_deployment").inc()
            # 记录详细错误日志
            logger.error(f"Stop deployment error for {model_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Stop deployment failed: {str(e)}"}), 500

@app.route("/metrics")
def metrics():
    """Prometheus指标接口"""
    # 记录请求计数，但不记录响应时间（避免递归问题）
    REQUEST_COUNT.labels(endpoint="/metrics", model_id="global").inc()
    
    try:
        return generate_latest(), 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        # 增加错误计数
        model_errors_counter.labels(model_id="global", error_type="metrics").inc()
        # 记录详细错误日志
        logger.error(f"Failed to generate metrics: {str(e)}", exc_info=True)
        return "Error generating metrics", 500, {'Content-Type': 'text/plain'}

@app.route("/health")
def health():
    """健康检查接口"""
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/health", model_id="global").inc()
    
    # 执行健康检查并计时
    with RESPONSE_TIME.labels(endpoint="/health", model_id="global").time():
        try:
            # 更新活跃服务数量指标
            update_active_services_metric()
            
            active_count = sum(1 for _, process_info in ml_service.model_processes.items() 
                              if process_info.get('status') == 'running')
            
            # 可选：执行自检
            health_status = {
                "status": "healthy",
                "service": "model-service",
                "timestamp": time.time(),
                "models": {
                    "total": len(ml_service.model_registry),
                    "deployed": active_count
                }
            }
            
            return jsonify(health_status), 200
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id="global", error_type="health").inc()
            # 记录详细错误日志
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"Health check failed: {str(e)}"}), 500

@app.route("/models/resources")
def models_resources():
    """获取所有已部署模型的资源使用情况"""
    # 记录请求指标
    REQUEST_COUNT.labels(endpoint="/models/resources", model_id="global").inc()
    
    # 执行资源查询并计时
    with RESPONSE_TIME.labels(endpoint="/models/resources", model_id="global").time():
        try:
            # 获取模型状态信息
            status_result, _ = ml_service.get_models_status()
            models_info = status_result.get('models', {})
            
            # 准备资源使用情况响应
            resources_data = {}
            
            # 遍历所有模型
            for model_id, model_info in models_info.items():
                # 只收集已部署的模型资源
                if model_info.get('deployment', {}).get('deployed', False):
                    # 获取进程信息
                    pid = None
                    process_info = ml_service.model_processes.get(model_id, {})
                    process = process_info.get('process')
                    port = process_info.get('port')
                    
                    # 检查进程是否在运行
                    is_process_running = False
                    if process:
                        try:
                            # 如果是Popen对象或其他类型，尝试获取PID
                            if hasattr(process, 'pid'):
                                pid = process.pid
                                
                                # 尝试验证PID是否有效
                                try:
                                    # 创建新的psutil.Process对象从pid
                                    psutil_process = psutil.Process(pid)
                                    # 确认进程存在
                                    psutil_process.status()
                                    is_process_running = True
                                    # 使用新的psutil对象替换原始process
                                    process = psutil_process
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    is_process_running = False
                        except (AttributeError, TypeError):
                            is_process_running = False
                    
                    if is_process_running:
                        try:
                            # 收集CPU使用率
                            cpu_percent = process.cpu_percent(interval=0.1)
                            
                            # 收集内存使用情况
                            memory_info = process.memory_info()
                            memory_usage_bytes = memory_info.rss
                            memory_percent = process.memory_percent()
                            
                            # 收集网络IO (如果可用)
                            net_io = None
                            net_io_rate = 0.0
                            
                            try:
                                io_counters = process.io_counters()
                                net_io = {
                                    'read_bytes': io_counters.read_bytes,
                                    'write_bytes': io_counters.write_bytes
                                }
                            except:
                                # 某些系统可能不支持io_counters
                                pass
                            
                            # 将资源信息添加到响应
                            resources_data[model_id] = {
                                'cpu_usage_percent': cpu_percent,
                                'memory_usage_bytes': memory_usage_bytes,
                                'memory_usage_percent': memory_percent,
                                'network_io': net_io,
                                'network_io_rate': net_io_rate,
                                'pid': pid,
                                'port': port,
                                'uptime': time.time() - process.create_time()
                            }
                        except Exception as e:
                            # 如果无法获取详细资源信息，只报告基本信息
                            resources_data[model_id] = {
                                'pid': pid,
                                'error': str(e),
                                'port': port
                            }
                    else:
                        # 模型已部署但进程可能已停止
                        resources_data[model_id] = {
                            'status': 'not_running',
                            'port': port
                        }
            
            # 添加系统总体资源使用情况
            resources_data['system'] = {
                'cpu_usage_percent': psutil.cpu_percent(),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'memory_available_bytes': psutil.virtual_memory().available
            }
            
            return jsonify({
                'status': 'success',
                'data': resources_data,
                'timestamp': time.time()
            }), 200
            
        except Exception as e:
            # 增加错误计数
            model_errors_counter.labels(model_id="global", error_type="resources").inc()
            # 记录详细错误日志
            logger.error(f"Failed to get models resources: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f"Failed to get models resources: {str(e)}"
            }), 500

@app.route("/debug/processes")
def debug_processes():
    """临时调试端点：查看模型进程信息"""
    try:
        result = {}
        for model_id, process_info in ml_service.model_processes.items():
            process = process_info.get('process')
            result[model_id] = {
                'pid': process.pid if hasattr(process, 'pid') else None,
                'status': process_info.get('status'),
                'port': process_info.get('port'),
                'has_pid': hasattr(process, 'pid') if process else False,
                'process_type': str(type(process)) if process else 'None'
            }
        
        # 添加一些系统信息
        result['system_info'] = {
            'python_version': sys.version,
            'psutil_version': psutil.__version__
        }
        
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error in debug endpoint: {str(e)}"
        }), 500

def run_registry(port):
    """在独立线程中运行注册中心服务"""
    run_registry_service(port=port)

def main():
    """应用程序入口"""
    parser = argparse.ArgumentParser(description='Model Service')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--registry_port', type=int, default=5050, help='Registry service port')
    args = parser.parse_args()
    
    global registry_thread
    
    try:
        # 启动注册服务
        registry_thread = threading.Thread(target=run_registry, args=(args.registry_port,))
        registry_thread.daemon = True
        registry_thread.start()
        logger.info(f"Registry service starting on port {args.registry_port}")
        
        # 启动模型资源监控
        start_metrics_collection()
        logger.info("Model metrics collector started")
        
        # 启动主应用
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Stopping services...")
        stop_metrics_collection()
    except Exception as e:
        logger.error(f"Error starting service: {str(e)}")
        stop_metrics_collection()
        sys.exit(1)

if __name__ == "__main__":
    main()
