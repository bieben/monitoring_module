import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_upload_models():
    """Test uploading multiple models"""
    print("\n🔄 Testing model uploads...")
    
    models = ["model1", "model2"]
    results = []
    
    for model_id in models:
        with open('test_model.pkl', 'rb') as f:
            files = {'file': f}
            data = {'model_id': model_id}
            response = requests.post(f"{BASE_URL}/upload_model", files=files, data=data)
            print(f"\nUploading {model_id}:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            results.append(response.status_code == 200)
    
    return all(results)

def test_predictions():
    """Test predictions with multiple models"""
    print("\n🔄 Testing predictions...")
    
    # California housing features
    test_data = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
    results = []
    
    for model_id in ["model1", "model2"]:
        data = {
            "model_id": model_id,
            "features": test_data
        }
        
        print(f"\nPredicting with {model_id}:")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        results.append(response.status_code == 200)
    
    return all(results)

def test_models_status():
    """Test models status endpoint"""
    print("\n🔄 Testing models status...")
    
    response = requests.get(f"{BASE_URL}/models/status")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()[0]  # Get the data part
        print("\n📊 Models Status:")
        print(f"Total Models: {data['total_models']}")
        
        for model_id, info in data['models'].items():
            print(f"\n🤖 Model: {model_id}")
            print("  Metadata:")
            print(f"    Upload Time: {info['metadata']['upload_time']}")
            print(f"    Features: {info['metadata']['feature_count']}")
            print("  Performance:")
            print(f"    Total Predictions: {info['performance']['total_predictions']}")
            print(f"    Average Latency: {info['performance']['avg_latency_ms']}ms")
            print(f"    Last Prediction: {info['performance']['last_prediction']}")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_metrics():
    """Test metrics endpoint"""
    print("\n🔄 Testing metrics...")
    
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print("Metrics available" if response.status_code == 200 else "Metrics not available")
    return response.status_code == 200

if __name__ == "__main__":
    # Wait for service to start
    print("Waiting for service to start...")
    time.sleep(2)
    
    # Run tests
    upload_success = test_upload_models()
    if upload_success:
        prediction_success = test_predictions()
        status_success = test_models_status()
    else:
        print("❌ Skipping prediction and status tests due to upload failure")
        prediction_success = False
        status_success = False
    
    metrics_success = test_metrics()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Model Uploads: {'✅' if upload_success else '❌'}")
    print(f"Predictions: {'✅' if prediction_success else '❌'}")
    print(f"Models Status: {'✅' if status_success else '❌'}")
    print(f"Metrics: {'✅' if metrics_success else '❌'}") 