from kafka import KafkaProducer, KafkaConsumer
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable

class KafkaClient:
    def __init__(self, bootstrap_servers: str = 'kafka:9092', log_callback: Optional[Callable] = None):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[KafkaProducer] = None
        self.log_callback = log_callback
        self._start_consumer()

    def get_producer(self) -> Optional[KafkaProducer]:
        if not self.producer:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
            except Exception as e:
                logging.error(f"Failed to connect to Kafka producer: {e}")
        return self.producer

    def send_log(self, log_data: Dict[str, Any]) -> bool:
        """发送日志到Kafka (向后兼容)"""
        return self.send_message("inference-logs", log_data)
        
    def send_message(self, topic: str, message_data: Dict[str, Any]) -> bool:
        """
        发送消息到指定的Kafka主题
        
        Args:
            topic: Kafka主题
            message_data: 要发送的消息数据
            
        Returns:
            是否发送成功
        """
        producer = self.get_producer()
        if not producer:
            return False
            
        try:
            future = producer.send(topic, message_data)
            future.get(timeout=10)
            return True
        except Exception as e:
            logging.error(f"Failed to send message to Kafka topic '{topic}': {e}")
            return False

    def _consume_logs(self):
        """消费日志消息并处理"""
        while True:
            try:
                consumer = KafkaConsumer(
                    'inference-logs',
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id='model-service-monitor',
                    auto_offset_reset='earliest'
                )

                for msg in consumer:
                    log = msg.value
                    
                    # 标准化日志格式
                    if isinstance(log, dict):
                        # 添加日志级别字段（如果没有）
                        if 'level' not in log:
                            # 基于潜在的错误或延迟情况推断日志级别
                            if 'error' in log or log.get('status_code', 200) >= 400:
                                log['level'] = 'ERROR'
                            elif log.get('latency', 0) > 1.0:
                                log['level'] = 'WARNING'
                            else:
                                log['level'] = 'INFO'
                        
                        # 添加时间戳（如果没有）
                        if 'timestamp' not in log:
                            log['timestamp'] = time.time()
                        
                        # 添加消息字段（如果没有）
                        if 'message' not in log:
                            if log.get('latency', 0) > 1.0:
                                log['message'] = f"High latency prediction: {log.get('latency')}s"
                            elif 'error' in log:
                                log['message'] = f"Error: {log.get('error')}"
                            else:
                                log['message'] = f"Prediction for model {log.get('model_id', 'unknown')}"
                    
                    # 处理高延迟情况的日志
                    if log.get('latency', 0) > 1.0:
                        logging.warning(f"High latency detected: {log.get('latency')}s for model {log.get('model_id', 'unknown')}")
                    
                    # 如果定义了回调函数，调用它处理日志
                    if self.log_callback:
                        try:
                            self.log_callback(log)
                        except Exception as e:
                            logging.error(f"Error in log callback: {e}")
                    
            except Exception as e:
                logging.error(f"Kafka consumer error: {e}")
                time.sleep(5)

    def _start_consumer(self):
        """启动日志消费者线程"""
        thread = threading.Thread(target=self._consume_logs, daemon=True)
        thread.start() 