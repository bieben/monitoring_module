import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 创建一个非常复杂的模型，会消耗大量CPU
X = np.random.random((1000, 100))
y = np.random.randint(0, 2, 1000)

# 创建一个包含100棵树且最大深度为20的随机森林模型
model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
model.fit(X, y)

# 创建特征名称
feature_names = [f'feature_{i}' for i in range(100)]

# 创建一个CPU密集型的预测函数
def cpu_intensive_predict(self, X):
    # 原始预测
    predictions = self.predict(X)
    
    # 额外CPU负载
    for _ in range(10000000):
        pass
    
    return predictions

# 替换模型的predict方法
model.original_predict = model.predict
model.predict = cpu_intensive_predict.__get__(model, RandomForestClassifier)

# 保存模型到文件
model_data = {
    'model': model,
    'feature_names': feature_names
}

# 确保models目录存在
os.makedirs('model-service/models', exist_ok=True)

# 保存model
joblib.dump(model_data, 'cpu_heavy_model.pkl')

print('已创建CPU密集型模型: cpu_heavy_model.pkl') 