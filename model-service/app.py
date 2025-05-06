from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
import logging
from service import MLService
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # 启用 CORS

# Prometheus metrics
REQUEST_COUNT = Counter('model_inference_requests_total', 'Total inference requests',
                       ['model_id'])  # Add model_id label
INFERENCE_LATENCY = Histogram('model_inference_latency_seconds', 'Inference latency in seconds',
                            ['model_id'])  # Add model_id label

# Initialize service
ml_service = MLService()

# Initialize metrics for existing models
for model_id in ml_service.model_registry.keys():
    REQUEST_COUNT.labels(model_id=model_id)
    INFERENCE_LATENCY.labels(model_id=model_id)

@app.route("/upload_model", methods=["POST"])
def upload_model():
    model_id = request.form.get("model_id")
    result, status_code = ml_service.handle_model_upload(model_id, request.files.get("file"))
    
    if status_code == 200:  # Only initialize metrics if upload was successful
        REQUEST_COUNT.labels(model_id=model_id)
        INFERENCE_LATENCY.labels(model_id=model_id)
    
    return jsonify(result), status_code

@app.route("/deploy", methods=["POST"])
def deploy():
    data = request.get_json()
    model_id = data.get("model_id")
    environment = data.get("environment", "development")

    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400

    config = {
        "environment": environment,
        "resources": {
            "cpu": "2 cores",
            "memory": "8GB",
            "gpu": "N/A"
        }
    }

    return jsonify(*ml_service.deploy_model(model_id, config))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data and "model_id" in data:
        REQUEST_COUNT.labels(model_id=data["model_id"]).inc()
        with INFERENCE_LATENCY.labels(model_id=data["model_id"]).time():
            return jsonify(*ml_service.handle_prediction(data))
    return jsonify({"error": "Invalid request"}), 400

@app.route("/models/status")
def models_status():
    """Get status of all registered models"""
    return jsonify(*ml_service.get_models_status())

@app.route("/delete_model/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """Delete a model"""
    return jsonify(*ml_service.delete_model(model_id))

@app.route("/stop_deployment/<model_id>", methods=["POST"])
def stop_deployment(model_id):
    """Stop a model deployment without deleting the model"""
    return jsonify(*ml_service.stop_deployment(model_id))

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

@app.route("/monitoring/config", methods=["GET"])
def get_monitoring_config():
    """获取监控配置"""
    category = request.args.get('category')
    key = request.args.get('key')
    config = ml_service.alert_rules.get_config(category, key)
    return jsonify(config)

@app.route("/monitoring/config/<category>", methods=["PUT"])
def update_monitoring_config(category):
    """更新监控配置"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        if 'key' in data and 'value' in data:
            # 更新单个配置项
            success = ml_service.alert_rules.update_config(category, data['key'], data['value'])
        else:
            # 更新整个类别
            success = ml_service.alert_rules.update_category(category, data)

        if success:
            return jsonify({"message": "Configuration updated successfully"})
        else:
            return jsonify({"error": "Failed to update configuration"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/monitoring/config/reset", methods=["POST"])
def reset_monitoring_config():
    """重置监控配置"""
    category = request.args.get('category')
    success = ml_service.alert_rules.reset_config(category)
    if success:
        return jsonify({"message": "Configuration reset successfully"})
    else:
        return jsonify({"error": "Failed to reset configuration"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
