#!/usr/bin/env python3
import os
import json
import time
import logging
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-registry')

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 模型注册表，存储模型ID到服务信息的映射
model_registry = {}
registry_lock = threading.RLock()  # 用于线程安全操作

# 端口映射文件目录
PORT_MAPPINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'port_mappings')
os.makedirs(PORT_MAPPINGS_DIR, exist_ok=True)

# 健康检查间隔（秒）
HEALTH_CHECK_INTERVAL = 30
MAX_FAILED_CHECKS = 3
failed_health_checks = {}

def load_registry_from_disk():
    """从磁盘加载注册表数据"""
    try:
        registry_file = os.path.join(PORT_MAPPINGS_DIR, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                data = json.load(f)
                with registry_lock:
                    model_registry.update(data)
                logger.info(f"Loaded {len(data)} model services from registry file")
                
        # 从端口映射文件加载
        for filename in os.listdir(PORT_MAPPINGS_DIR):
            if filename.endswith('.port'):
                model_id = filename[:-5]  # 去除.port后缀
                with open(os.path.join(PORT_MAPPINGS_DIR, filename), 'r') as f:
                    try:
                        port = int(f.read().strip())
                        with registry_lock:
                            if model_id not in model_registry:
                                model_registry[model_id] = {
                                    'port': port,
                                    'status': 'unknown',
                                    'last_updated': time.time(),
                                    'url': f"http://localhost:{port}"
                                }
                    except ValueError:
                        logger.warning(f"Invalid port in file {filename}")
    except Exception as e:
        logger.error(f"Error loading registry: {str(e)}")

def save_registry_to_disk():
    """将注册表数据保存到磁盘"""
    try:
        registry_file = os.path.join(PORT_MAPPINGS_DIR, 'registry.json')
        with registry_lock:
            with open(registry_file, 'w') as f:
                json.dump(model_registry, f, indent=2)
        logger.info(f"Saved registry with {len(model_registry)} entries to disk")
    except Exception as e:
        logger.error(f"Error saving registry: {str(e)}")

def update_port_mapping_file(model_id, port):
    """更新端口映射文件"""
    try:
        port_file = os.path.join(PORT_MAPPINGS_DIR, f"{model_id}.port")
        with open(port_file, 'w') as f:
            f.write(str(port))
    except Exception as e:
        logger.error(f"Error updating port mapping file for {model_id}: {str(e)}")

def check_service_health(model_id, url):
    """检查模型服务健康状态"""
    import requests
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            return True, status_data.get('status', 'unknown')
        return False, f"Health check failed with status {response.status_code}"
    except Exception as e:
        return False, f"Health check error: {str(e)}"

def run_health_checks():
    """定期运行健康检查"""
    while True:
        time.sleep(HEALTH_CHECK_INTERVAL)
        with registry_lock:
            models_to_check = list(model_registry.keys())
        
        for model_id in models_to_check:
            with registry_lock:
                if model_id not in model_registry:
                    continue
                service_info = model_registry[model_id]
                url = service_info.get('url')
            
            if not url:
                continue
                
            healthy, status = check_service_health(model_id, url)
            with registry_lock:
                if model_id not in model_registry:
                    continue
                    
                if healthy:
                    model_registry[model_id]['status'] = status
                    model_registry[model_id]['last_updated'] = time.time()
                    if model_id in failed_health_checks:
                        del failed_health_checks[model_id]
                else:
                    # 记录失败的健康检查
                    if model_id not in failed_health_checks:
                        failed_health_checks[model_id] = 0
                    failed_health_checks[model_id] += 1
                    
                    # 如果连续失败超过阈值，标记为不可用
                    if failed_health_checks[model_id] >= MAX_FAILED_CHECKS:
                        model_registry[model_id]['status'] = 'unavailable'
                        logger.warning(f"Model service {model_id} marked as unavailable after {MAX_FAILED_CHECKS} failed health checks")
                        
        # 每次健康检查后保存注册表
        save_registry_to_disk()

@app.route('/register', methods=['POST'])
def register_service():
    """注册模型服务"""
    data = request.get_json()
    if not data or 'model_id' not in data or 'port' not in data:
        return jsonify({'error': 'Missing required fields: model_id and port'}), 400
    
    model_id = data['model_id']
    port = data['port']
    
    # 构建服务URL（本地或远程）
    host = request.remote_addr if request.remote_addr != '127.0.0.1' else 'localhost'
    service_url = f"http://{host}:{port}"
    
    with registry_lock:
        model_registry[model_id] = {
            'model_id': model_id,
            'port': port,
            'url': service_url,
            'status': data.get('status', 'running'),
            'metadata': data.get('metadata', {}),
            'features': data.get('features', []),
            'registered_at': time.time(),
            'last_updated': time.time()
        }
    
    # 更新端口映射文件
    update_port_mapping_file(model_id, port)
    
    # 保存注册表
    save_registry_to_disk()
    
    logger.info(f"Registered model service {model_id} at {service_url}")
    return jsonify({
        'message': f"Service for model {model_id} registered successfully",
        'service_url': service_url
    }), 200

@app.route('/unregister/<model_id>', methods=['POST'])
def unregister_service(model_id):
    """注销模型服务"""
    with registry_lock:
        if model_id not in model_registry:
            return jsonify({'error': f"Model service {model_id} not found"}), 404
        
        service_info = model_registry.pop(model_id)
    
    # 删除端口映射文件
    port_file = os.path.join(PORT_MAPPINGS_DIR, f"{model_id}.port")
    if os.path.exists(port_file):
        os.remove(port_file)
    
    # 如果有失败的健康检查记录，清除它
    if model_id in failed_health_checks:
        del failed_health_checks[model_id]
    
    # 保存注册表
    save_registry_to_disk()
    
    logger.info(f"Unregistered model service {model_id}")
    return jsonify({'message': f"Service for model {model_id} unregistered successfully"}), 200

@app.route('/services', methods=['GET'])
def list_services():
    """列出所有注册的模型服务"""
    with registry_lock:
        services = {model_id: info.copy() for model_id, info in model_registry.items()}
    
    return jsonify({
        'services': services,
        'count': len(services)
    }), 200

@app.route('/service/<model_id>', methods=['GET'])
def get_service(model_id):
    """获取指定模型服务的信息"""
    with registry_lock:
        if model_id not in model_registry:
            return jsonify({'error': f"Model service {model_id} not found"}), 404
        
        service_info = model_registry[model_id].copy()
    
    return jsonify(service_info), 200

@app.route('/health', methods=['GET'])
def health_check():
    """注册表服务健康检查"""
    return jsonify({
        'status': 'healthy',
        'services_count': len(model_registry),
        'timestamp': time.time()
    }), 200

def run_registry_service(host='0.0.0.0', port=5050):
    """运行注册表服务"""
    # 从磁盘加载注册表
    load_registry_from_disk()
    
    # 启动健康检查线程
    health_check_thread = threading.Thread(target=run_health_checks, daemon=True)
    health_check_thread.start()
    
    # 启动Flask应用
    logger.info(f"Starting model registry service on port {port}")
    app.run(host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the model registry service')
    parser.add_argument('--port', type=int, default=5050, help='Port to run the registry service on')
    
    args = parser.parse_args()
    run_registry_service(port=args.port) 