import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from magilearn.models import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成回归数据集
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 使用训练集进行模型训练
model.fit(X_train, y_train)

# 使用模型对测试集进行预测
y_pred = model.predict(X_test)

# 计算均方误差和R²分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出模型评估指标
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")