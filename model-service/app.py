from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from kafka import KafkaProducer, KafkaConsumer
import json, time, uuid, logging, threading
from model import HousePriceModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_inference_requests_total', 'Total inference requests')
INFERENCE_LATENCY = Histogram('model_inference_latency_seconds', 'Inference latency in seconds')

# 初始化模型
model = HousePriceModel()

# Kafka producer with lazy init
producer = None

def get_producer():
    global producer
    if producer is None:
        for i in range(10):
            try:
                producer = KafkaProducer(
                    bootstrap_servers='kafka:9092',
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logging.info("✅ Kafka producer connected.")
                break
            except Exception as e:
                logging.warning(f"❌ Kafka producer connect failed (attempt {i+1}/10): {e}")
                time.sleep(5)
    return producer

@app.route("/predict", methods=["POST"])
@INFERENCE_LATENCY.time()
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        # 获取输入数据
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid input format"}), 400

        # 模型预测
        prediction = model.predict(data)
        latency = round(time.time() - start_time, 4)
        
        result = {
            "prediction": prediction,
            "latency": latency
        }

        # 发送日志到Kafka
        log_data = {
            "model_id": "california-housing",
            "request_id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "features": data,
            "prediction": prediction,
            "latency": latency,
            "status": "success"
        }

        kafka_producer = get_producer()
        if kafka_producer:
            try:
                future = kafka_producer.send("inference-logs", log_data)
                record_metadata = future.get(timeout=10)
                logging.info(f"✅ Message sent to topic={record_metadata.topic} partition={record_metadata.partition} offset={record_metadata.offset}")
            except Exception as e:
                logging.warning(f"❌ Kafka send failed: {e}")

        return jsonify(result)

    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ Prediction error: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

# Kafka consumer in background thread
def consume_logs():
    connected = False
    while not connected:
        logging.info("🔄 Trying to connect Kafka consumer...")
        try:
            consumer = KafkaConsumer(
                'inference-logs',
                bootstrap_servers='kafka:9092',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='model-service-monitor-new',
                request_timeout_ms=30000,
                session_timeout_ms=10000
            )

            logging.info("✅ Kafka consumer connected and started.")
            connected = True

            for msg in consumer:
                logging.info("🔥 Message received from topic: %s, partition: %d, offset: %d", 
                           msg.topic, msg.partition, msg.offset)
                log = msg.value
                latency = log.get("latency", 0)
                logging.info(f"📥 Consumed: model={log['model_id']}, prediction={log['prediction']:.2f}, latency={latency}s")
                if latency > 1.0:
                    logging.warning(f"🚨 High latency alert: {latency}s")

        except Exception as e:
            logging.warning(f"❌ Kafka consumer connection failed. Retrying in 5s... Error: {e}")
            time.sleep(5)

def start_consumer_thread():
    t = threading.Thread(target=consume_logs, daemon=True)
    t.start()

if __name__ == "__main__":
    start_consumer_thread()
    app.run(host="0.0.0.0", port=5000)
