import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from magilearn.models import RandomForestClassifier

# 创建一个简单的分类数据集:100 个样本、5 个特征的分类数据集，分为 3 个类别
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_clusters_per_class=1, n_informative=3, random_state=42)

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用 PCA 将数据降至二维
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# 实例化并训练 RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10, max_features=3)
rf_clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred_custom = rf_clf.predict(X_test)

# 计算自定义模型的准确率
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Custom RandomForest Accuracy: {accuracy_custom * 100:.2f}%")

# 实例化并训练 sklearn 中的 RandomForestClassifier
sklearn_rf_clf = SklearnRandomForest(n_estimators=10, max_features=3)
sklearn_rf_clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred_sklearn = sklearn_rf_clf.predict(X_test)

# 计算 sklearn 模型的准确率
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn RandomForest Accuracy: {accuracy_sklearn * 100:.2f}%")

# 混淆矩阵的可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自定义模型的混淆矩阵
cm_custom = confusion_matrix(y_test, y_pred_custom)
disp_custom = ConfusionMatrixDisplay(confusion_matrix=cm_custom)
disp_custom.plot(ax=ax[0])
ax[0].set_title("自定义 RandomForest Confusion Matrix")

# Sklearn 模型的混淆矩阵
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
disp_sklearn = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn)
disp_sklearn.plot(ax=ax[1])
ax[1].set_title("Sklearn RandomForest Confusion Matrix")

plt.tight_layout()
plt.show()

# 决策边界图（PCA降至二维时）
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 使用自定义模型进行预测
Z_custom = rf_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_custom = Z_custom.reshape(xx.shape)

# 使用sklearn模型进行预测
Z_sklearn = sklearn_rf_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_sklearn = Z_sklearn.reshape(xx.shape)

plt.figure(figsize=(12, 6))

# 自定义模型的决策边界
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_custom, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title("自定义 RandomForest - Decision Boundary")

# Sklearn模型的决策边界
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_sklearn, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Sklearn RandomForest - Decision Boundary")

plt.tight_layout()
plt.show()


