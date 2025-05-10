import requests
import threading
import time
import random
import concurrent.futures

# 模型参数
MODEL_ID = "test_model"
FEATURES = [8.3252, 41, 6.984, 1.023, 322, 2.555, 37.88, -122.23]
BASE_URL = "http://localhost:5000"
NUM_THREADS = 10
REQUESTS_PER_THREAD = 500

def send_request():
    """发送一个预测请求"""
    # 稍微修改特征值以避免缓存
    features = [f + random.random() * 0.1 for f in FEATURES]
    
    # 准备请求数据
    url = f"{BASE_URL}/predict"
    data = {
        "model_id": MODEL_ID,
        "features": features
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            print(f"Error: {response.status_code}: {response.text}")
            return False, None
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False, None

def worker(thread_id, results):
    """工作线程函数"""
    success_count = 0
    
    for i in range(REQUESTS_PER_THREAD):
        # 增加CPU计算负载 - 计算素数
        compute_intensive_operation()
        
        # 发送请求
        success, _ = send_request()
        if success:
            success_count += 1
        
        # 进度报告
        if (i + 1) % 50 == 0:
            print(f"线程 {thread_id}: 已完成 {i+1}/{REQUESTS_PER_THREAD} 请求")
    
    results[thread_id] = success_count

def compute_intensive_operation():
    """执行计算密集型操作，增加CPU负载"""
    # 计算前1000个素数
    primes = []
    for num in range(2, 1000):
        is_prime = all(num % i != 0 for i in range(2, int(num**0.5) + 1))
        if is_prime:
            primes.append(num)
    
    # 矩阵乘法
    a = [[random.random() for _ in range(50)] for _ in range(50)]
    b = [[random.random() for _ in range(50)] for _ in range(50)]
    result = [[sum(a[i][k] * b[k][j] for k in range(50)) for j in range(50)] for i in range(50)]
    
    return len(primes), result[0][0]

def concurrent_requests():
    """使用线程池并发发送请求"""
    print(f"使用线程池并发发送 {NUM_THREADS * REQUESTS_PER_THREAD} 个请求...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(send_request) for _ in range(NUM_THREADS * REQUESTS_PER_THREAD)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"已完成 {completed}/{NUM_THREADS * REQUESTS_PER_THREAD} 请求")
    
    duration = time.time() - start_time
    print(f"总共发送了 {NUM_THREADS * REQUESTS_PER_THREAD} 个请求，耗时 {duration:.2f} 秒")
    print(f"每秒请求数: {NUM_THREADS * REQUESTS_PER_THREAD / duration:.2f}")

def main():
    print(f"开始测试: {NUM_THREADS} 个线程，每个线程 {REQUESTS_PER_THREAD} 个请求")
    
    # 方法1: 使用单独的线程
    results = {}
    threads = []
    
    start_time = time.time()
    
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    duration = time.time() - start_time
    total_success = sum(results.values())
    
    print("\n=== 测试结果 ===")
    print(f"总请求数: {NUM_THREADS * REQUESTS_PER_THREAD}")
    print(f"成功请求数: {total_success}")
    print(f"失败请求数: {NUM_THREADS * REQUESTS_PER_THREAD - total_success}")
    print(f"耗时: {duration:.2f} 秒")
    print(f"每秒请求数: {NUM_THREADS * REQUESTS_PER_THREAD / duration:.2f}")
    
    # 检查告警状态
    time.sleep(2)  # 等待告警生成
    try:
        alerts_response = requests.get(f"{BASE_URL}/alerts")
        print("\n=== 当前告警状态 ===")
        print(alerts_response.json())
    except Exception as e:
        print(f"获取告警状态时出错: {str(e)}")

if __name__ == "__main__":
    main() 