# 模型服务微服务架构 API 使用指南

本文档提供了模型服务微服务架构的 API 使用示例，包括如何上传模型、部署服务、执行预测以及管理模型服务。

## API 概览

系统由三个主要组件组成：

1. **主服务** - 负责模型上传、部署和管理，运行在端口 5000
2. **注册中心** - 负责维护模型服务注册信息，运行在端口 5050
3. **独立模型服务** - 每个模型运行在独立进程，端口自动分配（通常从 8000 开始）

完整的 API 文档可以在 `swagger.yaml` 文件中找到。

## 使用示例

以下是使用 `curl` 和 Python 示例代码展示如何使用这些 API。

### 1. 上传模型

将训练好的模型上传到服务。模型应该是一个 `.pkl` 文件，包含一个字典，其中包含 'model' 和可选的 'feature_names' 键。

#### curl 示例：

```bash
curl -X POST http://localhost:5000/upload_model \
  -F "model_id=iris_model" \
  -F "file=@/path/to/iris_model.pkl"
```

#### Python 示例：

```python
import requests

url = "http://localhost:5000/upload_model"
model_id = "iris_model"
model_path = "/path/to/iris_model.pkl"

files = {
    "file": open(model_path, "rb")
}
data = {
    "model_id": model_id
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 2. 部署模型服务

将上传的模型部署为独立的微服务。

#### curl 示例：

```bash
curl -X POST http://localhost:5000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "iris_model",
    "environment": "production",
    "resources": {
      "cpu_limit": "2",
      "memory_limit": "512MB",
      "timeout": 60
    }
  }'
```

#### Python 示例：

```python
import requests

url = "http://localhost:5000/deploy"
data = {
    "model_id": "iris_model",
    "environment": "production",
    "resources": {
        "cpu_limit": "2",
        "memory_limit": "512MB",
        "timeout": 60
    }
}

response = requests.post(url, json=data)
print(response.json())
# 输出示例: {"message": "Model iris_model deployed successfully", "service_url": "http://localhost:8000", "port": 8000, ...}
```

### 3. 执行预测

使用模型进行预测。您可以通过主服务进行预测（将自动转发到独立服务），或直接访问独立服务。

#### 通过主服务（推荐）：

##### curl 示例：

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "iris_model",
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

##### Python 示例：

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "model_id": "iris_model",
    "features": [5.1, 3.5, 1.4, 0.2]
}

response = requests.post(url, json=data)
print(response.json())
# 输出示例: {"prediction": 0.0, "latency": 0.0021, "model_id": "iris_model", "timestamp": 1634567890.123}
```

#### 直接访问独立服务：

如果知道独立服务的端口（从部署响应中获取），可以直接访问它：

##### curl 示例：

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

##### Python 示例：

```python
import requests

# 假设我们知道模型服务运行在端口 8000
service_url = "http://localhost:8000"
data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

response = requests.post(f"{service_url}/infer", json=data)
print(response.json())
```

### 4. 获取模型状态

获取所有模型的状态信息。

#### curl 示例：

```bash
curl -X GET http://localhost:5000/models/status
```

#### Python 示例：

```python
import requests

url = "http://localhost:5000/models/status"
response = requests.get(url)
print(response.json())
```

### 5. 停止模型部署

停止特定模型的部署，但不删除模型文件。

#### curl 示例：

```bash
curl -X POST http://localhost:5000/stop_deployment/iris_model
```

#### Python 示例：

```python
import requests

model_id = "iris_model"
url = f"http://localhost:5000/stop_deployment/{model_id}"
response = requests.post(url)
print(response.json())
```

### 6. 删除模型

删除模型（包括停止其服务，如果正在运行）。

#### curl 示例：

```bash
curl -X DELETE http://localhost:5000/delete_model/iris_model
```

#### Python 示例：

```python
import requests

model_id = "iris_model"
url = f"http://localhost:5000/delete_model/{model_id}"
response = requests.delete(url)
print(response.json())
```

## 完整工作流示例

以下是一个完整的工作流示例，包括模型上传、部署、预测和清理。

```python
import requests
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 1. 准备模型
def create_sample_model():
    # 加载示例数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 训练一个简单的模型
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    # 保存模型
    model_data = {
        "model": model,
        "feature_names": iris.feature_names
    }
    
    joblib.dump(model_data, "iris_model.pkl")
    return "iris_model.pkl"

# 2. 上传模型
def upload_model(model_path, model_id):
    url = "http://localhost:5000/upload_model"
    files = {"file": open(model_path, "rb")}
    data = {"model_id": model_id}
    
    response = requests.post(url, files=files, data=data)
    print(f"上传模型: {response.json()}")
    return response.status_code == 200

# 3. 部署模型服务
def deploy_model(model_id):
    url = "http://localhost:5000/deploy"
    data = {
        "model_id": model_id,
        "environment": "production",
        "resources": {
            "cpu_limit": "2",
            "memory_limit": "512MB"
        }
    }
    
    response = requests.post(url, json=data)
    print(f"部署模型: {response.json()}")
    
    if response.status_code == 200:
        return response.json().get("port")
    return None

# 4. 测试预测
def test_prediction(model_id, features):
    # 通过主服务
    url = "http://localhost:5000/predict"
    data = {
        "model_id": model_id,
        "features": features
    }
    
    response = requests.post(url, json=data)
    print(f"通过主服务预测: {response.json()}")
    
    return response.json()

# 5. 清理
def cleanup(model_id):
    url = f"http://localhost:5000/delete_model/{model_id}"
    response = requests.delete(url)
    print(f"清理模型: {response.json()}")
    return response.status_code == 200

# 运行完整流程
def run_workflow():
    model_id = "test_iris_model"
    
    # 创建并上传模型
    model_path = create_sample_model()
    if not upload_model(model_path, model_id):
        print("模型上传失败")
        return
    
    # 部署模型
    port = deploy_model(model_id)
    if not port:
        print("模型部署失败")
        return
    
    # 等待服务启动
    time.sleep(2)
    
    # 测试预测
    sample_features = [5.1, 3.5, 1.4, 0.2]  # 样本特征
    prediction_result = test_prediction(model_id, sample_features)
    
    # 查看所有模型状态
    response = requests.get("http://localhost:5000/models/status")
    print(f"所有模型状态: {response.json()}")
    
    # 清理
    cleanup(model_id)

if __name__ == "__main__":
    run_workflow()
```

## 监控

系统提供了 Prometheus 兼容的监控指标，可以通过以下端点访问：

1. 主服务: `http://localhost:5000/metrics`
2. 独立模型服务: `http://localhost:<port>/metrics`

这些指标可以用于监控以下内容：

- 请求计数和延迟
- 预测错误率
- 活跃服务数量
- 内存使用情况

您可以使用 Prometheus 和 Grafana 等工具可视化这些指标。

## 错误处理

所有 API 在遇到错误时将返回适当的 HTTP 状态码和错误信息。常见的错误代码包括：

- 400: 请求参数错误
- 404: 模型或服务未找到
- 500: 服务器内部错误

错误响应示例：

```json
{
  "error": "Model not found. Please upload first."
}
```

## 高级配置

### 资源限制

在部署模型时，您可以配置资源限制：

```json
{
  "model_id": "my_model",
  "resources": {
    "cpu_limit": "4",
    "memory_limit": "1GB",
    "timeout": 120
  }
}
```

### 环境配置

可以为不同环境配置部署参数：

```json
{
  "model_id": "my_model",
  "environment": "development"
}
```

## 更多信息

有关 API 的完整规范，请参阅 `swagger.yaml` 文件。 