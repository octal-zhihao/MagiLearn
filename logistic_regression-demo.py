import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from magilearn.models import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 选择前两个特征进行可视化
y = (iris.target != 0).astype(int)  # 将多分类问题转化为二分类问题（非类别 0 为正类）

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 scikit-learn 的逻辑回归模型
model = LogisticRegression(learning_rate=0.001, num_iterations=10000, tol=1e-6)
# model = LogisticRegression(solver='lbfgs', max_iter=10000, tol=1e-6)
model.fit(X_train, y_train)  # 拟合模型

# 预测并评估模型
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of Logistic Regression")
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# # from magilearn.models import LogisticRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # 生成数据集
# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化逻辑回归模型
# model = LogisticRegression(solver='lbfgs', max_iter=10000, tol=1e-6)
# # model = LogisticRegression(learning_rate=0.01, num_iterations=10000, tol=1e-6)
# model.fit(X_train, y_train)  # 拟合模型

# # 预测并评估模型
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))


# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
# x1 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
# x2 = -(model.coef_[0] * x1 + model.intercept_) / model.coef_[1]
# plt.plot(x1, x2, color='red', label="Decision Boundary")
# plt.legend()
# plt.show()
