#!/usr/bin/env python3
import requests
import json
import time
import argparse
import os
from pathlib import Path
import logging
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-service-test')

class ModelServiceTest:
    def __init__(self, base_url="http://localhost:5000", registry_url="http://localhost:5050", container_mode=False):
        self.base_url = base_url
        self.registry_url = registry_url
        self.model_id = f"test_model_{int(time.time())}"
        self.test_model_path = "./test_model.pkl"
        self.deployed_port = None
        self.model_url = None
        self.container_mode = container_mode
        
        # 容器端口映射，如果在容器模式下使用
        self.port_mapping = {
            8000: 7000,  # 从容器内8000端口映射到主机7000端口
            8001: 7001,
            8002: 7002,
            8003: 7003,
            8004: 7004,
            8005: 7005,
            8006: 7006,
            8007: 7007,
            8008: 7008,
            8009: 7009,
            8010: 7010
        }

    def map_port(self, port):
        """在容器模式下映射端口"""
        if self.container_mode and port in self.port_mapping:
            return self.port_mapping[port]
        return port

    def run_all_tests(self):
        """运行所有测试"""
        try:
            logger.info("===== 开始测试模型服务 =====")
            
            # 1. 检查服务是否运行
            logger.info("1. 测试主服务健康状态")
            self.test_health()
            
            # 2. 检查注册中心是否运行
            logger.info("2. 测试注册中心健康状态")
            self.test_registry_health()
            
            # 3. 上传模型
            logger.info("3. 测试模型上传")
            self.test_upload_model()
            
            # 4. 部署模型
            logger.info("4. 测试模型部署")
            self.test_deploy_model()
            
            # 5. 获取模型状态
            logger.info("5. 测试获取模型状态")
            self.test_model_status()
            
            # 6. 直接向模型服务发送预测请求
            if self.model_url:
                logger.info("6. 测试直接向模型服务发送预测请求")
                self.test_direct_prediction()
            
            # 7. 通过主服务发送预测请求
            logger.info("7. 测试通过主服务发送预测请求")
            self.test_proxy_prediction()
            
            # 8. 验证模型服务的指标
            if self.model_url:
                logger.info("8. 测试模型服务指标")
                self.test_model_metrics()
            
            # 9. 停止模型部署
            logger.info("9. 测试停止模型部署")
            self.test_stop_deployment()
            
            # 10. 删除模型
            logger.info("10. 测试删除模型")
            self.test_delete_model()
            
            logger.info("===== 所有测试完成 =====")
            return True
        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
            return False

    def test_health(self):
        """测试主服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"主服务健康状态: {response.json()}")
                return True
            else:
                logger.error(f"主服务健康检查失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"主服务健康检查异常: {str(e)}")
            raise

    def test_registry_health(self):
        """测试注册中心健康状态"""
        try:
            response = requests.get(f"{self.registry_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"注册中心健康状态: {response.json()}")
                return True
            else:
                logger.error(f"注册中心健康检查失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"注册中心健康检查异常: {str(e)}")
            raise

    def test_upload_model(self):
        """测试模型上传"""
        try:
            # 检查测试模型文件是否存在
            if not os.path.exists(self.test_model_path):
                logger.error(f"测试模型文件不存在: {self.test_model_path}")
                raise FileNotFoundError(f"测试模型文件不存在: {self.test_model_path}")
            
            # 发送上传请求
            with open(self.test_model_path, 'rb') as f:
                files = {'file': (f"{self.model_id}.pkl", f, 'application/octet-stream')}
                data = {'model_id': self.model_id}
                
                response = requests.post(
                    f"{self.base_url}/upload_model",
                    files=files,
                    data=data,
                    timeout=10
                )
            
            if response.status_code == 200:
                logger.info(f"模型上传成功: {response.json()}")
                return True
            else:
                logger.error(f"模型上传失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"模型上传异常: {str(e)}")
            raise

    def test_deploy_model(self):
        """测试模型部署"""
        try:
            # 部署配置
            deploy_config = {
                "model_id": self.model_id,
                "environment": "testing",
                "resources": {
                    "cpu_limit": "1",
                    "memory_limit": "256MB",
                    "timeout": 30
                }
            }
            
            # 发送部署请求
            response = requests.post(
                f"{self.base_url}/deploy",
                json=deploy_config,
                timeout=30  # 部署可能需要更长时间
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"模型部署成功: {result}")
                
                # 保存部署信息
                if 'port' in result:
                    self.deployed_port = result['port']
                    
                    # 如果在容器模式下，需要映射端口
                    mapped_port = self.map_port(self.deployed_port)
                    
                    # 根据是否为容器模式，使用不同的URL
                    if self.container_mode:
                        self.model_url = f"http://localhost:{mapped_port}"
                        logger.info(f"容器模式: 将容器内端口 {self.deployed_port} 映射到主机端口 {mapped_port}")
                    else:
                        self.model_url = result.get('service_url', f"http://localhost:{self.deployed_port}")
                    
                    logger.info(f"模型服务URL: {self.model_url}")
                
                # 等待模型服务完全启动
                if self.model_url:
                    max_retries = 10
                    retry_interval = 1
                    for i in range(max_retries):
                        try:
                            health_response = requests.get(f"{self.model_url}/health", timeout=2)
                            if health_response.status_code == 200:
                                logger.info(f"模型服务就绪: {health_response.json()}")
                                break
                        except:
                            pass
                        
                        logger.info(f"等待模型服务就绪...({i+1}/{max_retries})")
                        time.sleep(retry_interval)
                
                return True
            else:
                logger.error(f"模型部署失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"模型部署异常: {str(e)}")
            raise

    def test_model_status(self):
        """测试获取模型状态"""
        try:
            response = requests.get(f"{self.base_url}/models/status", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"模型状态: {json.dumps(result, indent=2)}")
                
                # 检查我们的测试模型是否存在并已部署
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    models_info = result[0].get('models', {})
                elif isinstance(result, dict):
                    models_info = result.get('models', {})
                else:
                    logger.warning("无法识别的模型状态数据结构")
                    return False
                    
                if self.model_id in models_info:
                    model_info = models_info[self.model_id]
                    deployment_info = model_info.get('deployment', {})
                    
                    if deployment_info.get('deployed', False):
                        logger.info(f"测试模型已成功部署")
                    else:
                        logger.warning(f"测试模型未成功部署")
                else:
                    logger.warning(f"测试模型 {self.model_id} 不在状态列表中")
                
                return True
            else:
                logger.error(f"获取模型状态失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"获取模型状态异常: {str(e)}")
            raise

    def test_direct_prediction(self):
        """测试直接向模型服务发送预测请求"""
        if not self.model_url:
            logger.warning("模型URL未设置，跳过直接预测测试")
            return False
        
        try:
            # 构建测试数据
            payload = {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
            
            # 发送预测请求
            response = requests.post(
                f"{self.model_url}/infer",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"直接预测结果: {result}")
                return True
            else:
                logger.error(f"直接预测失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"直接预测异常: {str(e)}")
            raise

    def test_proxy_prediction(self):
        """测试通过主服务发送预测请求"""
        try:
            # 构建测试数据
            payload = {
                "model_id": self.model_id,
                "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
            
            # 发送预测请求
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"转发预测结果: {result}")
                return True
            else:
                logger.error(f"转发预测失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"转发预测异常: {str(e)}")
            raise

    def test_model_metrics(self):
        """测试模型服务指标"""
        if not self.model_url:
            logger.warning("模型URL未设置，跳过模型服务指标测试")
            return False
        
        try:
            response = requests.get(f"{self.model_url}/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = response.text
                logger.info(f"模型服务指标示例:\n{metrics[:500]}...(截断)")
                
                # 检查是否包含预期的指标
                expected_metrics = [
                    "model_predictions_total",
                    "model_prediction_latency_seconds",
                    "model_feature_count",
                    "model_loaded_timestamp"
                ]
                
                for metric in expected_metrics:
                    if metric in metrics:
                        logger.info(f"找到预期的指标: {metric}")
                    else:
                        logger.warning(f"未找到预期的指标: {metric}")
                
                return True
            else:
                logger.error(f"获取模型服务指标失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"获取模型服务指标异常: {str(e)}")
            raise

    def test_stop_deployment(self):
        """测试停止模型部署"""
        try:
            response = requests.post(
                f"{self.base_url}/stop_deployment/{self.model_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"停止模型部署成功: {response.json()}")
                
                # 验证服务是否已停止
                if self.model_url:
                    try:
                        health_response = requests.get(f"{self.model_url}/health", timeout=2)
                        if health_response.status_code == 200:
                            logger.warning(f"模型服务仍在运行: {health_response.json()}")
                        else:
                            logger.info("模型服务已停止")
                    except:
                        logger.info("模型服务已停止 (无法连接)")
                
                return True
            else:
                logger.error(f"停止模型部署失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"停止模型部署异常: {str(e)}")
            raise

    def test_delete_model(self):
        """测试删除模型"""
        try:
            response = requests.delete(
                f"{self.base_url}/delete_model/{self.model_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"删除模型成功: {response.json()}")
                return True
            else:
                logger.error(f"删除模型失败，状态码: {response.status_code}，响应: {response.text}")
                return False
        except Exception as e:
            logger.error(f"删除模型异常: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试模型服务API')
    parser.add_argument('--base-url', default='http://localhost:5000', help='主服务URL')
    parser.add_argument('--registry-url', default='http://localhost:5050', help='注册中心URL')
    parser.add_argument('--test-model', default='./test_model.pkl', help='测试模型文件路径')
    parser.add_argument('--container-mode', action='store_true', help='使用容器模式，端口映射')
    
    args = parser.parse_args()
    
    tester = ModelServiceTest(
        base_url=args.base_url,
        registry_url=args.registry_url,
        container_mode=args.container_mode
    )
    tester.test_model_path = args.test_model
    
    success = tester.run_all_tests()
    if success:
        logger.info("所有测试通过!")
        exit(0)
    else:
        logger.error("测试失败!")
        exit(1) 