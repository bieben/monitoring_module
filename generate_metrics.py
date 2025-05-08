#!/usr/bin/env python3
import requests
import time
import random
import json
import argparse
from datetime import datetime

def send_inference_request(url, model_id, data_size, verbose=False):
    """发送推理请求到model-service"""
    # 生成随机数据
    data = [random.random() for _ in range(data_size)]
    
    # 构建请求数据
    payload = {
        "model_id": model_id,
        "data": data
    }
    
    # 发送请求
    try:
        start_time = time.time()
        response = requests.post(
            url, 
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        elapsed = time.time() - start_time
        
        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Request to {model_id} with {data_size} items: Status {response.status_code}, Time {elapsed:.4f}s")
        
        return response.status_code == 200
    except Exception as e:
        if verbose:
            print(f"Error: {str(e)}")
        return False

def generate_load(base_url, duration_minutes, requests_per_minute, verbose=False):
    """生成负载"""
    # 可选的模型ID
    model_ids = ["model1", "model2", "test_model"]
    
    # 可选的数据大小
    data_sizes = [5, 10, 20, 50, 100]
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # 请求计数
    request_count = 0
    success_count = 0
    
    print(f"开始生成负载，将持续 {duration_minutes} 分钟...")
    
    while time.time() < end_time:
        # 计算本批次请求数量
        interval = 60 / requests_per_minute
        batch_size = 1
        
        for _ in range(batch_size):
            # 随机选择模型ID和数据大小
            model_id = random.choice(model_ids)
            data_size = random.choice(data_sizes)
            
            # 发送请求
            success = send_inference_request(f"{base_url}/predict", model_id, data_size, verbose)
            request_count += 1
            if success:
                success_count += 1
        
        # 等待一定时间后发送下一批请求
        time.sleep(interval)
    
    # 打印统计信息
    elapsed = time.time() - start_time
    print(f"\n负载生成完成！")
    print(f"总运行时间: {elapsed:.2f} 秒")
    print(f"总请求数: {request_count}")
    print(f"成功请求数: {success_count}")
    print(f"成功率: {(success_count/request_count)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate load for model-service")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the model-service")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    parser.add_argument("--rpm", type=int, default=30, help="Requests per minute")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    generate_load(args.url, args.duration, args.rpm, args.verbose)
