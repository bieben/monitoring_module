import logging
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import numpy as np

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self._train()

    def _train(self):
        """训练模型"""
        logging.info("🔄 Loading and training model...")
        data = fetch_california_housing()
        self.model = LinearRegression()
        self.model.fit(data.data, data.target)
        self.feature_names = data.feature_names
        logging.info("✅ Model training completed")

    def predict(self, features: dict) -> float:
        """
        预测房价
        
        Args:
            features (dict): 特征字典，键为特征名，值为特征值
            
        Returns:
            float: 预测的房价（单位：十万美元）
        """
        # 构建特征向量
        feature_vector = []
        for feature in self.feature_names:
            if feature not in features:
                raise ValueError(f"Missing feature: {feature}")
            feature_vector.append(float(features[feature]))

        # 预测
        prediction = float(self.model.predict([feature_vector])[0])
        return prediction

    def get_feature_names(self) -> list:
        """获取特征名列表"""
        return self.feature_names 