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

def run_registry():
    """在独立线程中运行注册中心服务"""
    registry_port = int(os.environ.get('REGISTRY_PORT', '5050'))
    run_registry_service(port=registry_port)

def main():
    parser = argparse.ArgumentParser(description='启动模型服务')
    parser.add_argument('--port', type=int, default=5000, help='主服务端口')
    parser.add_argument('--registry-port', type=int, default=5050, help='注册中心端口')
    parser.add_argument('--no-registry', action='store_true', help='不启动内置的注册中心')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['REGISTRY_PORT'] = str(args.registry_port)
    
    # 启动注册中心服务（在独立线程中）
    if not args.no_registry:
        logger.info(f"Starting model registry service on port {args.registry_port}")
        registry_thread = threading.Thread(target=run_registry, daemon=True)
        registry_thread.start()
    
    # 启动主服务
    logger.info(f"Starting model service on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
