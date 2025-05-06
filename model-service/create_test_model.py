from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import joblib

# Load and train model
data = fetch_california_housing()
model = LinearRegression()
model.fit(data.data, data.target)

# Save model
joblib.dump({
    'model': model,
    'feature_names': data.feature_names
}, 'test_model.pkl')

print("✅ Test model created successfully") 