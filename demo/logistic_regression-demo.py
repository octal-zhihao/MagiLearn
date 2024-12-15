import numpy as np
import matplotlib.pyplot as plt
from magilearn.datasets import load_iris, train_test_split
from magilearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from magilearn.models import LogisticRegression

# 加载鸢尾花数据集
iris = load_iris()
X = iris['data'][:, :2]  # 选择前两个特征进行可视化
y = (iris['target'] != 0).astype(int)  # 将多分类问题转化为二分类问题（非类别 0 为正类）

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# 使用自定义的逻辑回归模型
custom_model = LogisticRegression(learning_rate=0.001, num_iterations=10000, tol=1e-6)
custom_model.fit(X_train, y_train)

# 使用 sklearn 的逻辑回归模型
sklearn_model = SklearnLogisticRegression(solver='lbfgs', max_iter=10000, tol=1e-6)
sklearn_model.fit(X_train, y_train)

# 预测并评估模型
y_pred_custom = custom_model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)

# 输出准确率
print("Magilearn Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Sklearn Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_sklearn))

# 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测决策边界
Z_custom = custom_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_custom = Z_custom.reshape(xx.shape)

Z_sklearn = sklearn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_sklearn = Z_sklearn.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(12, 6))

# 自定义模型的决策边界
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_custom, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Magilearn Logistic Regression - Decision Boundary")

# Sklearn模型的决策边界
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_sklearn, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Sklearn Logistic Regression - Decision Boundary")

plt.tight_layout()
plt.show()
