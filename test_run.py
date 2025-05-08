#!/usr/bin/env python3
"""
简单测试脚本，用于验证预测分析模块是否能够运行
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 导入预测分析模块
    from predictive_analytics.src.predictive_analyzer import PredictiveAnalyzer
    from predictive_analytics.src.collectors.base_collector import BaseCollector
    
    logger.info("模块导入成功")
    
    # 创建一个模拟的数据收集器
    class MockCollector(BaseCollector):
        def __init__(self):
            super().__init__({})
            
        def collect_metrics(self, start_time, end_time):
            """生成模拟数据"""
            logger.info(f"收集从 {start_time} 到 {end_time} 的指标")
            
            dates = pd.date_range(
                start=start_time,
                end=end_time,
                freq='5min'
            )
            
            # 创建模拟数据
            data = pd.DataFrame({
                'timestamp': dates,
                'cpu_usage': np.random.uniform(10, 80, size=len(dates)),
                'memory_usage': np.random.uniform(100, 1000, size=len(dates)),
                'network_io': np.random.uniform(10, 500, size=len(dates)),
                'latency': np.random.uniform(0.1, 0.8, size=len(dates))
            })
            
            logger.info(f"生成了 {len(data)} 行模拟数据")
            return data
    
    # 创建预测分析器
    logger.info("创建预测分析器...")
    analyzer = PredictiveAnalyzer(
        data_collector=MockCollector(),
        model_cache_path="./models/test_cache",
        prediction_horizon=15  # 15分钟预测
    )
    
    # 收集最近12小时的数据
    logger.info("收集数据...")
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=12)
    data = analyzer.collect_metrics(start_time, end_time)
    
    logger.info(f"数据收集和预处理完成，形状: {data.shape}")
    
    # 进行预测
    logger.info("使用Prophet模型预测资源使用...")
    predictions = analyzer.predict_resources(data, use_prophet=True)
    
    logger.info("预测结果：")
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            logger.info(f"{key}: 形状={value.shape}, 均值={np.mean(value):.2f}")
    
    # 优化资源分配
    logger.info("优化资源分配...")
    constraints = {
        'max_cpu': 100,  # 百分比
        'max_memory': 4 * 1024,  # MB
        'max_network': 1000,  # MB/s
        'max_latency': 1.0,  # 秒
        'optimization_objective': 'cost',  # 成本优化
        'safety_margin': 1.2  # 20%安全边际
    }
    
    optimization = analyzer.optimize_allocation(predictions, constraints)
    
    logger.info("优化结果：")
    logger.info(f"CPU分配: {optimization['cpu_allocation']:.2f}")
    logger.info(f"内存分配: {optimization['memory_allocation']:.2f}")
    logger.info(f"网络分配: {optimization['network_allocation']:.2f}")
    logger.info(f"状态: {optimization['status']}")
    
    # 导出结果
    logger.info("导出结果到JSON...")
    export_data = analyzer.export_results(predictions, optimization, export_format='json')
    
    logger.info("测试完成，预测分析模块工作正常！")
    
except Exception as e:
    logger.error(f"运行过程中出现错误: {str(e)}", exc_info=True)
    
if __name__ == "__main__":
    pass  # 已在脚本顶部执行了所有代码 