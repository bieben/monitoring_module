#!/usr/bin/env python3
"""
查询并展示各个模型的资源使用情况的命令行工具
"""

import requests
import json
import argparse
import time
import sys
from tabulate import tabulate
import os

def get_model_resources(service_url='http://localhost:5000'):
    """获取模型资源使用情况"""
    try:
        response = requests.get(f"{service_url}/models/resources", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to model service: {str(e)}")
        return None

def format_bytes(size_bytes):
    """格式化字节大小为人类可读格式"""
    if size_bytes is None:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_seconds(seconds):
    """格式化秒数为人类可读时间格式"""
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"

def display_resources(resources_data, format="table"):
    """展示资源使用情况"""
    if not resources_data or 'data' not in resources_data:
        print("No resource data available")
        return
    
    data = resources_data['data']
    system_info = data.pop('system', {}) if 'system' in data else {}
    timestamp = resources_data.get('timestamp', time.time())
    
    # 系统总体信息
    print("\n=== System Resources ===")
    if system_info:
        print(f"CPU Usage: {system_info.get('cpu_usage_percent', 'N/A'):.2f}%")
        print(f"Memory Usage: {system_info.get('memory_usage_percent', 'N/A'):.2f}%")
        print(f"Available Memory: {format_bytes(system_info.get('memory_available_bytes'))}")
    else:
        print("No system information available")
    
    # 模型资源信息
    if not data:
        print("\nNo active models found")
        return
    
    print(f"\n=== Model Resources ({len(data)} models) ===")
    
    if format == "table":
        # 表格格式
        table_data = []
        headers = ["Model ID", "Status", "CPU %", "Memory", "Memory %", "Network I/O", "Uptime"]
        
        for model_id, resources in data.items():
            status = resources.get('status', 'running')
            if status == 'not_running':
                row = [model_id, "STOPPED", "N/A", "N/A", "N/A", "N/A", "N/A"]
            elif 'error' in resources:
                row = [model_id, "ERROR", "N/A", "N/A", "N/A", "N/A", "N/A"]
            else:
                memory = format_bytes(resources.get('memory_usage_bytes'))
                memory_percent = f"{resources.get('memory_usage_percent', 0):.2f}%"
                cpu = f"{resources.get('cpu_usage_percent', 0):.2f}%"
                
                # 网络I/O
                net_io = resources.get('network_io', {})
                if net_io:
                    network = f"{format_bytes(net_io.get('read_bytes', 0))} in / {format_bytes(net_io.get('write_bytes', 0))} out"
                else:
                    network = "N/A"
                
                uptime = format_seconds(resources.get('uptime'))
                row = [model_id, "RUNNING", cpu, memory, memory_percent, network, uptime]
            
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    else:
        # JSON格式
        print(json.dumps(data, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Show model resource usage')
    parser.add_argument('--url', default='http://localhost:5000', help='Model service URL')
    parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    parser.add_argument('--watch', action='store_true', help='Continuously monitor resources')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds for watch mode')
    
    args = parser.parse_args()
    
    try:
        if args.watch:
            # 持续监控模式
            print(f"Monitoring models resources (Ctrl+C to exit). Refreshing every {args.interval} seconds...")
            try:
                while True:
                    # 清屏
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # 显示当前时间
                    print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 获取并显示资源
                    resources = get_model_resources(args.url)
                    if resources:
                        display_resources(resources, args.format)
                    
                    # 等待刷新间隔
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        else:
            # 单次查询模式
            resources = get_model_resources(args.url)
            if resources:
                display_resources(resources, args.format)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 