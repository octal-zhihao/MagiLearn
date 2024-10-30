"""
A Simplified Machine Learning Library
=====================================

magi-learn integrates basic machine learning algorithms and preprocessing tools
to provide simple and efficient solutions to common learning problems.

See the documentation for more details.
"""

# 导入模型
from .models.logistic_regression import LogisticRegression
from .models.linear_regression import LinearRegression
from .models.k_means import KMeans

# 导入预处理工具
from .preprocessing.scaler import StandardScaler
from .preprocessing.encoder import OneHotEncoder

# 导入评估指标
from .metrics.accuracy import accuracy_score
from .metrics.mean_squared_error import mean_squared_error

# 导入工具模块
from .utils.data_split import train_test_split

# 定义公开接口
__all__ = [
    "LogisticRegression",
    "LinearRegression",
    "KMeans",
    "StandardScaler",
    "OneHotEncoder",
    "accuracy_score",
    "mean_squared_error",
    "train_test_split",
]
