import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from magilearn.models import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# 生成回归数据集
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化自定义的线性回归模型
custom_model = LinearRegression(n_iters=100000)

# 使用训练集进行自定义模型训练
custom_model.fit(X_train, y_train)

# 使用自定义模型对测试集进行预测
y_pred_custom = custom_model.predict(X_test)

# 计算自定义模型的均方误差和R²分数
mse_custom = mean_squared_error(y_test, y_pred_custom)
r2_custom = r2_score(y_test, y_pred_custom)

# 初始化sklearn的线性回归模型
sklearn_model = SklearnLinearRegression()

# 使用训练集进行sklearn模型训练
sklearn_model.fit(X_train, y_train)

# 使用sklearn模型对测试集进行预测
y_pred_sklearn = sklearn_model.predict(X_test)

# 计算sklearn模型的均方误差和R²分数
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

# 输出模型评估指标
print(f"Magilearn Linear Regression - Mean Squared Error: {mse_custom:.4f}")
print(f"Magilearn Linear Regression - R² Score: {r2_custom:.4f}")
print(f"Sklearn Linear Regression - Mean Squared Error: {mse_sklearn:.4f}")
print(f"Sklearn Linear Regression - R² Score: {r2_sklearn:.4f}")

# 可视化对比图
plt.figure(figsize=(14, 6))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制实际数据点与自定义模型的回归线
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='实际数据')
plt.plot(X_test, y_pred_custom, color='red', label='自定义模型预测')
plt.title('Magilearn Linear Regression - 实际 vs 预测')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

# 绘制实际数据点与sklearn模型的回归线
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='实际数据')
plt.plot(X_test, y_pred_sklearn, color='green', label='Sklearn模型预测')
plt.title('Sklearn Linear Regression - 实际 vs 预测')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

plt.tight_layout()
plt.show()