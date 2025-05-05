from kafka import KafkaProducer, KafkaConsumer
import json
import logging
import threading
import time
from typing import Optional

class KafkaClient:
    def __init__(self, bootstrap_servers: str = 'kafka:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[KafkaProducer] = None
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

    def send_log(self, log_data: dict) -> bool:
        producer = self.get_producer()
        if not producer:
            return False
            
        try:
            future = producer.send("inference-logs", log_data)
            future.get(timeout=10)
            return True
        except Exception as e:
            logging.error(f"Failed to send log to Kafka: {e}")
            return False

    def _consume_logs(self):
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
                    if log.get("latency", 0) > 1.0:
                        logging.warning(f"High latency detected: {log.get('latency')}s")
                    
            except Exception as e:
                logging.error(f"Kafka consumer error: {e}")
                time.sleep(5)

    def _start_consumer(self):
        thread = threading.Thread(target=self._consume_logs, daemon=True)
        thread.start() 