import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SklearnKMeans
from magilearn.models import KMeans
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 指定使用 4 个核心

# 生成聚类数据集
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# 划分训练集和测试集
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# 实例化并训练自定义 KMeans 模型
model_custom = KMeans(n_clusters=3, random_state=42)
model_custom.fit(X_train)

# 对测试集进行预测
y_pred_custom = model_custom.predict(X_test)

# 计算自定义模型的轮廓系数
silhouette_custom = silhouette_score(X_test, y_pred_custom)

# 输出自定义 KMeans 的轮廓系数和聚类中心
print(f"Magilearn KMeans Silhouette Score: {silhouette_custom:.4f}")
print("Magilearn KMeans Cluster Centers:\n", model_custom.cluster_centers_)

# 实例化并训练 sklearn 的 KMeans 模型
model_sklearn = SklearnKMeans(n_clusters=3, random_state=42)
model_sklearn.fit(X_train)

# 对测试集进行预测
y_pred_sklearn = model_sklearn.predict(X_test)

# 计算 sklearn 模型的轮廓系数
silhouette_sklearn = silhouette_score(X_test, y_pred_sklearn)

# 输出 sklearn KMeans 的轮廓系数和聚类中心
print(f"Sklearn KMeans Silhouette Score: {silhouette_sklearn:.4f}")
print("Sklearn KMeans Cluster Centers:\n", model_sklearn.cluster_centers_)

# 可视化对比
plt.figure(figsize=(12, 6))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自定义 KMeans 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_custom, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.scatter(model_custom.cluster_centers_[:, 0], model_custom.cluster_centers_[:, 1], color='red', marker='x', s=100, label='Centroids')
plt.title(f"Magilearn KMeans - Silhouette Score: {silhouette_custom:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Sklearn KMeans 聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_sklearn, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.scatter(model_sklearn.cluster_centers_[:, 0], model_sklearn.cluster_centers_[:, 1], color='red', marker='x', s=100, label='Centroids')
plt.title(f"Sklearn KMeans - Silhouette Score: {silhouette_sklearn:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()