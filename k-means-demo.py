import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans
from magilearn.models import KMeans

# 生成聚类数据集
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# 划分训练集和测试集
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# 初始化k-means模型，设置聚类中心数量为3
model = KMeans(n_clusters=3, random_state=42)

# 使用训练集进行模型训练
model.fit(X_train)

# 使用模型对测试集进行预测
y_pred = model.predict(X_test)

# 计算轮廓系数
silhouette = silhouette_score(X_test, y_pred)

# 输出模型评估指标
print(f"Silhouette Score: {silhouette:.4f}")

# 输出聚类中心
print("Cluster Centers:\n", model.cluster_centers_)
