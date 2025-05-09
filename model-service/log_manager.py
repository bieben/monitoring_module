from typing import List, Dict, Any, Optional
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque


class LogManager:
    """管理系统日志，提供日志缓存、查询和清理功能"""
    
    def __init__(self, max_cache_size: int = 10000):
        """
        初始化日志管理器
        
        Args:
            max_cache_size: 内存中保留的最大日志条数
        """
        self.logs = deque(maxlen=max_cache_size)  # 使用双端队列存储日志
        self.max_cache_size = max_cache_size
        self.lock = threading.RLock()  # 线程安全的锁
        self._start_cleanup_thread()
        logging.info(f"LogManager initialized with max cache size: {max_cache_size}")
    
    def add_log(self, log_data: Dict[str, Any]):
        """
        添加新日志到内存缓存
        
        Args:
            log_data: 日志数据，字典格式
        """
        # 确保log_data包含所需的字段
        if 'timestamp' not in log_data:
            log_data['timestamp'] = time.time()
            
        if 'level' not in log_data:
            log_data['level'] = 'INFO'
        
        with self.lock:
            self.logs.append(log_data)
    
    def query_logs(self, 
                  limit: int = 50, 
                  offset: int = 0, 
                  level: Optional[str] = None,
                  model_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询日志
        
        Args:
            limit: 返回的最大日志条数
            offset: 分页偏移量
            level: 日志级别过滤
            model_id: 模型ID过滤
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            query: 消息内容搜索
            
        Returns:
            符合条件的日志列表
        """
        with self.lock:
            # 过滤日志
            filtered_logs = list(self.logs)
            
            if level:
                filtered_logs = [log for log in filtered_logs if log.get('level', '').upper() == level.upper()]
                
            if model_id:
                filtered_logs = [log for log in filtered_logs if log.get('model_id') == model_id]
                
            if start_time:
                filtered_logs = [log for log in filtered_logs if log.get('timestamp', 0) >= start_time]
                
            if end_time:
                filtered_logs = [log for log in filtered_logs if log.get('timestamp', 0) <= end_time]
                
            if query:
                query = query.lower()
                filtered_logs = [log for log in filtered_logs if 
                               query in str(log.get('message', '')).lower() or 
                               query in str(log.get('context', {})).lower()]
            
            # 按时间戳倒序排序（最新的日志在前）
            filtered_logs.sort(key=lambda log: log.get('timestamp', 0), reverse=True)
            
            # 分页
            total = len(filtered_logs)
            paginated_logs = filtered_logs[offset:offset + limit] if offset < total else []
            
            # 处理日志格式，确保timestamp是ISO格式的字符串
            for log in paginated_logs:
                if 'timestamp' in log and isinstance(log['timestamp'], (int, float)):
                    # 转换为ISO格式的字符串
                    log['timestamp'] = datetime.fromtimestamp(
                        log['timestamp'], tz=timezone.utc
                    ).isoformat()
            
            return {
                "logs": paginated_logs,
                "total": total,
                "offset": offset,
                "limit": limit,
                "filters_active": {
                    "level": level,
                    "model_id": model_id,
                    "time_range": [start_time, end_time]
                }
            }
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        获取日志摘要统计信息
        
        Returns:
            日志统计信息
        """
        with self.lock:
            total_logs = len(self.logs)
            
            # 计算每个级别的日志数量
            level_counts = {}
            for log in self.logs:
                level = log.get('level', 'UNKNOWN').upper()
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # 计算每个模型的日志数量
            model_counts = {}
            for log in self.logs:
                model_id = log.get('model_id', 'unknown')
                model_counts[model_id] = model_counts.get(model_id, 0) + 1
            
            # 找出最早和最晚的日志时间
            timestamps = []
            for log in self.logs:
                ts = log.get('timestamp', 0)
                # 确保时间戳是数字类型
                if isinstance(ts, str):
                    try:
                        # 尝试解析ISO格式的字符串时间戳
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        ts = dt.timestamp()
                    except (ValueError, TypeError):
                        ts = 0
                timestamps.append(ts)
                
            earliest = min(timestamps) if timestamps else 0
            latest = max(timestamps) if timestamps else 0
            
            return {
                "total_logs": total_logs,
                "level_distribution": level_counts,
                "model_distribution": model_counts,
                "time_range": {
                    "earliest": datetime.fromtimestamp(earliest, tz=timezone.utc).isoformat() if earliest else None,
                    "latest": datetime.fromtimestamp(latest, tz=timezone.utc).isoformat() if latest else None
                },
                "cache_usage": f"{total_logs}/{self.max_cache_size}"
            }
    
    def clear_logs(self):
        """清空日志缓存"""
        with self.lock:
            self.logs.clear()
            logging.info("Log cache cleared")
    
    def _cleanup_old_logs(self):
        """清理过期日志的后台任务"""
        # 当前简单实现，仅依赖deque的maxlen自动清理
        # 未来可以基于时间等更复杂的策略
        pass
        
    def _start_cleanup_thread(self):
        """启动日志清理线程"""
        cleanup_thread = threading.Thread(
            target=self._cleanup_thread_func,
            daemon=True  # 设为守护线程，主线程结束时自动退出
        )
        cleanup_thread.start()
        
    def _cleanup_thread_func(self):
        """日志清理线程函数"""
        while True:
            try:
                # 每小时执行一次清理
                time.sleep(3600)
                self._cleanup_old_logs()
            except Exception as e:
                logging.error(f"Error in log cleanup thread: {e}") 