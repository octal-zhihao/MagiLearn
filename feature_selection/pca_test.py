import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
from pca import PCA

# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建 PCA 对象，指定要保留的主成分数
pca = PCA(n_components=2)

# 拟合模型并转换数据
X_pca = pca.fit_transform(X)

# 绘制 PCA 结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Species')
plt.grid()
plt.show()
