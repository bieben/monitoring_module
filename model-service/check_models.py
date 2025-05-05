import requests
import json
import os
from datetime import datetime

# Get service URL from environment or use default
SERVICE_URL = os.getenv('MODEL_SERVICE_URL', 'http://localhost:5000')

def format_time(time_str):
    """Format time string to be more readable"""
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

def check_models_status():
    """Check and display status of all models"""
    try:
        response = requests.get(f'{SERVICE_URL}/models/status')
        if response.status_code != 200:
            print(f"❌ Error: {response.json()}")
            return

        data = response.json()[0]
        print(data)
        # Print header
        print("\n" + "="*50)
        print(f"📊 Model Service Status Report")
        print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Service URL: {SERVICE_URL}")
        print(f"📚 Total Models: {data['total_models']}")
        print("="*50)

        # Print each model's status
        for model_id, info in data['models'].items():
            print(f"\n🤖 Model: {model_id}")
            print("  └─ Status:", "🟢" if info['status'] == 'active' else "🔴")
            
            # Metadata
            print("  └─ Metadata:")
            print(f"     ├─ Upload Time: {format_time(info['metadata']['upload_time'])}")
            print(f"     ├─ Feature Count: {info['metadata']['feature_count']}")
            print(f"     └─ Features: {', '.join(info['metadata']['feature_names'])}")
            
            # Performance
            print("  └─ Performance:")
            print(f"     ├─ Total Predictions: {info['performance']['total_predictions']}")
            print(f"     ├─ Average Latency: {info['performance']['avg_latency_ms']:.2f}ms")
            print(f"     └─ Last Prediction: {info['performance']['last_prediction']}")
            print("   " + "-"*40)
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to the model service at {SERVICE_URL}")
        print("  Is the service running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    check_models_status() 