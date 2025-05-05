import logging
import os
import joblib
import time
import uuid
from typing import Dict, Any, Tuple
from kafka_client import KafkaClient
from flask import jsonify
class MLService:
    def __init__(self):
        self.model_registry: Dict[str, Dict[str, Any]] = {}  # model_id -> {model, metadata}
        self.kafka_client = KafkaClient()
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        # Sync model registry with files on startup
        self._sync_model_registry()

    def _sync_model_registry(self):
        """Synchronize model registry with model files in the models directory"""
        # Clear current registry
        self.model_registry.clear()
        
        # List all .pkl files in models directory
        try:
            model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
            for model_file in model_files:
                model_id = model_file[:-4]  # Remove .pkl extension
                try:
                    model_path = os.path.join("models", model_file)
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.model_registry[model_id] = {
                            'model': model_data['model'],
                            'feature_names': model_data.get('feature_names', []),
                            'upload_time': os.path.getmtime(model_path),
                            'prediction_count': 0,
                            'avg_latency': 0.0
                        }
                        logging.info(f"Loaded model {model_id} from {model_path}")
                    else:
                        logging.warning(f"Skipped invalid model file: {model_file}")
                except Exception as e:
                    logging.error(f"Failed to load model {model_file}: {e}")
        except Exception as e:
            logging.error(f"Failed to sync model registry: {e}")

    def _check_model_file_exists(self, model_id: str) -> bool:
        """Check if model file exists"""
        model_path = os.path.join("models", f"{model_id}.pkl")
        return os.path.exists(model_path)

    def handle_model_upload(self, model_id: str, model_file) -> Tuple[dict, int]:
        """Handle model upload request"""
        if not model_id or not model_file:
            return {"error": "Missing model_id or file"}, 400

        try:
            save_path = os.path.join("models", f"{model_id}.pkl")
            model_file.save(save_path)

            # Load model data
            model_data = joblib.load(save_path)
            if not isinstance(model_data, dict) or 'model' not in model_data:
                os.remove(save_path)  # Clean up invalid file
                return {"error": "Invalid model format"}, 400
            
            # Store model with metadata
            self.model_registry[model_id] = {
                'model': model_data['model'],
                'feature_names': model_data.get('feature_names', []),
                'upload_time': time.time(),
                'prediction_count': 0,
                'avg_latency': 0.0
            }
            return {"message": f"Model {model_id} uploaded successfully"}, 200
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)  # Clean up on error
            logging.error(f"Model upload failed: {e}")
            return {"error": str(e)}, 500
        
    def deploy_model(self, model_id: str, config: Dict[str, Any]) -> Tuple[dict, int]:
        """
        Deploy a model for prediction with specified configuration.
        In minimal case, we treat it as activating the model in memory.
        """
        if not model_id:
            return {"error": "Missing model_id"}, 400

        if model_id not in self.model_registry:
            return {"error": "Model not found. Please upload first."}, 404

        try:
            model_info = self.model_registry[model_id]

            # Simulate deployment configuration (can be extended)
            model_info['deployed'] = True
            model_info['deploy_config'] = config
            model_info['deploy_time'] = time.time()

            logging.info(self.model_registry[model_id])

            return {"message": f"Model {model_id} deployed successfully",
                    "config": config}, 200
        except Exception as e:
            logging.error(f"Deployment failed for model {model_id}: {e}")
            return {"error": str(e)}, 500


    def handle_prediction(self, data: dict) -> Tuple[dict, int]:
        """Handle prediction request"""
        if not data or "model_id" not in data or "features" not in data:
            return {"error": "Missing model_id or features"}, 400

        model_id = data["model_id"]
        
        # Check if model exists in both registry and file system
        if not self._check_model_file_exists(model_id):
            if model_id in self.model_registry:
                del self.model_registry[model_id]  # Remove from registry if file is missing
            return {"error": "Model file not found"}, 404

        start_time = time.time()
        try:
            # Get model and metadata
            model_info = self.model_registry.get(model_id)
            if not model_info:
                return {"error": "Model not registered"}, 404

            model = model_info['model']
            prediction = float(model.predict([data["features"]])[0])
            latency = time.time() - start_time

            # Update model statistics
            model_info['prediction_count'] += 1
            model_info['avg_latency'] = (
                (model_info['avg_latency'] * (model_info['prediction_count'] - 1) + latency)
                / model_info['prediction_count']
            )

            result = {
                "prediction": prediction,
                "latency": latency,
                "model_stats": {
                    "total_predictions": model_info['prediction_count'],
                    "avg_latency": model_info['avg_latency']
                }
            }

            self._log_prediction(model_id, prediction, latency, data["features"])
            return result, 200
        except Exception as e:
            logging.error(f"Prediction failed for model {model_id}: {e}")
            return {"error": str(e)}, 500

    def delete_model(self, model_id: str) -> Tuple[dict, int]:
        """Delete a model from both file system and registry"""
        if not model_id:
            return {"error": "Missing model_id"}, 400

        try:
            model_path = os.path.join("models", f"{model_id}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
                if model_id in self.model_registry:
                    del self.model_registry[model_id]
                return {"message": f"Model {model_id} deleted successfully"}, 200
            else:
                if model_id in self.model_registry:
                    del self.model_registry[model_id]
                return {"error": "Model file not found"}, 404
        except Exception as e:
            logging.error(f"Failed to delete model {model_id}: {e}")
            return {"error": str(e)}, 500

    def stop_deployment(self, model_id: str) -> Tuple[dict, int]:
        """Stop a model deployment without deleting the model file"""
        if not model_id:
            return {"error": "Missing model_id"}, 400

        try:
            if model_id not in self.model_registry:
                return {"error": "Model not found"}, 404

            # Update model info to reflect stopped deployment
            model_info = self.model_registry[model_id]
            model_info['deployed'] = False
            if 'deploy_config' in model_info:
                del model_info['deploy_config']
            if 'deploy_time' in model_info:
                del model_info['deploy_time']

            return {"message": f"Model {model_id} deployment stopped successfully"}, 200
        except Exception as e:
            logging.error(f"Failed to stop deployment for model {model_id}: {e}")
            return {"error": str(e)}, 500

    def get_models_status(self) -> Tuple[dict, int]:
        """Get status of all registered models"""
        try:
            
            status = {}
            for model_id, info in self.model_registry.items():
                status[model_id] = {
                    'status': 'active' if info.get('deployed') else 'inactive',
                    'metadata': {
                        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                   time.localtime(info['upload_time'])),
                        'feature_names': info['feature_names'],
                        'feature_count': len(info['feature_names'])
                    },
                    'performance': {
                        'total_predictions': info['prediction_count'],
                        'avg_latency_ms': round(info['avg_latency'] * 1000, 2),  # Convert to milliseconds
                        'last_prediction': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                       time.localtime(time.time())) if info['prediction_count'] > 0 else 'Never'
                    },
                    'deployment': {
                        'deployed': info.get('deployed', False),
                        'deploy_config': info.get('deploy_config', {}),
                        'deploy_time': time.strftime('%Y-%m-%d %H:%M:%S',
                                                    time.localtime(info['deploy_time']))
                        if 'deploy_time' in info else 'Not deployed'
                    }
                }
            return {
                "models": status,
                "total_models": len(status)
            }, 200
        except Exception as e:
            logging.error(f"Failed to get models status: {e}")
            return {"error": str(e)}, 500

    def _log_prediction(self, model_id: str, prediction: float, latency: float, features: list):
        """Log prediction details"""
        log_data = {
            "model_id": model_id,
            "request_id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "features": features,
            "prediction": prediction,
            "latency": latency,
            "status": "success"
        }
        self.kafka_client.send_log(log_data) 