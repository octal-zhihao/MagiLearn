import numpy as np
from magilearn.models import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 生成模拟的回归数据集
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建岭回归模型，设置正则化参数alpha
ridge = Ridge(alpha=1.0)

# 在训练集上训练模型
ridge.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = ridge.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印评估结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R² Score): {r2:.4f}")

# 可视化真实值和预测值
plt.scatter(X_test, y_test, color='blue', label='真实值')
plt.plot(X_test, y_pred, color='red', label='预测值')
plt.title('岭回归：真实值与预测值')
plt.xlabel('特征')
plt.ylabel('目标值')
plt.legend()
plt.show()
