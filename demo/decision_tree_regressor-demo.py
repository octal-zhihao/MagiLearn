import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from magilearn.models import DecisionTreeRegressor  # 自定义回归器
from sklearn.tree import DecisionTreeRegressor as SklearnDTR  # sklearn 的回归器

# 1. 生成模拟的回归数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 特征数据
noise = np.random.normal(0, 0.5, size=100)  # 添加噪声
y = np.sin(X).ravel() + noise  # 目标值

# 2. 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建两个回归器
# 自定义的回归器
my_regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
my_regressor.fit(X_train, y_train)

# sklearn 提供的回归器
sklearn_regressor = SklearnDTR(max_depth=5, min_samples_split=5)
sklearn_regressor.fit(X_train, y_train)

# 4. 在测试集上进行预测
my_y_pred = my_regressor.predict(X_test)
sklearn_y_pred = sklearn_regressor.predict(X_test)

# 5. 评价两款模型的性能
my_mse = mean_squared_error(y_test, my_y_pred)
sklearn_mse = mean_squared_error(y_test, sklearn_y_pred)

print(f"\n测试集上的自定义回归器 MSE: {my_mse:.4f}")
print(f"测试集上的 sklearn 回归器 MSE: {sklearn_mse:.4f}\n")

# 6. 可视化两个模型的比较
plt.figure(figsize=(12, 6))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练集的真实值
plt.scatter(X_train, y_train, color="blue", label="训练数据", alpha=0.6)

# 测试集的真实值
plt.scatter(X_test, y_test, color="green", label="测试数据", alpha=0.7)

# 自定义回归器的预测值
X_sorted = np.sort(X_test, axis=0)  # 排序 X 便于绘图
my_y_pred_sorted = my_regressor.predict(X_sorted)
plt.plot(X_sorted, my_y_pred_sorted, color="red", label="Magilearn 回归器", linewidth=2)

# sklearn 回归器的预测值
sklearn_y_pred_sorted = sklearn_regressor.predict(X_sorted)
plt.plot(X_sorted, sklearn_y_pred_sorted, color="orange", label="sklearn 回归器", linewidth=2)

# 图片设置
plt.title("Decision Tree Regressor Comparison")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
