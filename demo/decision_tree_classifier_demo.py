import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from magilearn.models.decision_tree_classifier import DecisionTreeClassifier

# 1. 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 只保留两类数据 (Setosa 和 Versicolor)，将其简化为二分类
binary_mask = (y == 0) | (y == 1)
X, y = X[binary_mask], y[binary_mask]

# 只取前两个特征方便可视化
X = X[:, :2]

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 使用自定义的决策树分类器训练并预测
custom_clf = DecisionTreeClassifier(max_depth=3, min_samples_split=4)
custom_clf.fit(X_train, y_train)
y_pred_custom = custom_clf.predict(X_test)

# 4. 使用 sklearn 的决策树分类器训练并预测
sklearn_clf = SklearnDecisionTree(max_depth=3, min_samples_split=4)
sklearn_clf.fit(X_train, y_train)
y_pred_sklearn = sklearn_clf.predict(X_test)

# 5. 输出准确率
print(f"Magilearn决策树准确率: {accuracy_score(y_test, y_pred_custom):.2f}")
print(f"Sklearn决策树准确率: {accuracy_score(y_test, y_pred_sklearn):.2f}")

# 6. 可视化预测结果
plt.figure(figsize=(12, 6))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制自定义决策树的预测结果
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='black',
            s=100, label='真实值', alpha=0.7, facecolors='none')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=50,
            label='预测值', linewidths=2, alpha=0.7, color='red')
plt.title('Magilearn决策树分类器预测结果')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper right')

# 绘制sklearn决策树的预测结果
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='black',
            s=100, label='真实值', alpha=0.7, facecolors='none')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=50,
            label='预测值', linewidths=2, alpha=0.7, color='red')
plt.title('Sklearn决策树分类器预测结果')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
