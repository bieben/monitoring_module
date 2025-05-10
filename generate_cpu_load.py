import requests
import threading
import time
import random
import sys

# 全局参数
MODEL_ID = "test_model"
NUM_THREADS = 20
NUM_REQUESTS_PER_THREAD = 500
BASE_URL = "http://localhost:5000"  # 主服务的URL
MODEL_URL = "http://localhost:8001"  # 直接访问模型服务

# 创建一个CPU密集型的函数，在请求中计算大量素数
def compute_primes(n):
    primes = []
    for num in range(2, n + 1):
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                break
        else:
            primes.append(num)
    return primes

def make_request(thread_id, direct=False):
    """发送预测请求"""
    # 创建随机特征数据
    features = [random.random() for _ in range(8)]
    
    # 增加CPU负载 - 计算素数
    compute_primes(1000)
    
    # 准备请求数据
    if direct:
        url = f"{MODEL_URL}/infer"
        data = {"features": features}
    else:
        url = f"{BASE_URL}/predict"
        data = {"model_id": MODEL_ID, "features": features}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=10)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            return True, elapsed
        else:
            print(f"Thread {thread_id}: Error - {response.status_code}: {response.text}")
            return False, elapsed
    except Exception as e:
        print(f"Thread {thread_id}: Exception - {str(e)}")
        return False, 0

def worker(thread_id, results, direct=False):
    """工作线程函数"""
    success_count = 0
    total_time = 0
    
    for i in range(NUM_REQUESTS_PER_THREAD):
        success, elapsed = make_request(thread_id, direct)
        if success:
            success_count += 1
            total_time += elapsed
        
        # 进度报告
        if (i + 1) % 50 == 0:
            print(f"Thread {thread_id}: Completed {i+1}/{NUM_REQUESTS_PER_THREAD} requests")
    
    # 存储结果
    results[thread_id] = (success_count, total_time)

def main():
    # 是否直接访问模型服务
    direct = len(sys.argv) > 1 and sys.argv[1].lower() == "direct"
    
    print(f"Starting load test with {NUM_THREADS} threads, {NUM_REQUESTS_PER_THREAD} requests per thread")
    print(f"Mode: {'Direct to model service' if direct else 'Via main service'}")
    
    # 创建线程
    threads = []
    results = {}
    
    start_time = time.time()
    
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(i, results, direct))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 计算统计数据
    total_time = time.time() - start_time
    total_requests = NUM_THREADS * NUM_REQUESTS_PER_THREAD
    successful_requests = sum(res[0] for res in results.values())
    avg_latency = sum(res[1] for res in results.values()) / successful_requests if successful_requests > 0 else 0
    
    print("\n=== Load Test Results ===")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests} ({successful_requests/total_requests*100:.2f}%)")
    print(f"Failed requests: {total_requests - successful_requests}")
    print(f"Average latency: {avg_latency*1000:.2f} ms")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {total_requests/total_time:.2f}")

if __name__ == "__main__":
    main() 