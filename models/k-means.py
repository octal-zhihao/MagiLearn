import numpy as np

class KMeans:
    """
    KMeans 聚类模型，用于无监督学习的聚类任务。
    
    该实现通过随机初始化质心并迭代优化的方法，将数据划分为 K 个簇。

    参数：
        n_clusters (int): 聚类数量，默认为 8。
        max_iters (int): 最大迭代次数，默认为 300。
        tol (float): 收敛阈值，默认为 1e-4。
    
    属性：
        centers (numpy.ndarray): 每个簇的中心点坐标。
        labels (numpy.ndarray): 每个样本的簇标签。

    方法：
        fit(X): 使用训练数据对模型进行训练。
        predict(X): 对新数据进行聚类预测。
    """

    def __init__(self, n_clusters=8, max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        self.labels = None

    def _initialize_centers(self, X):
        """
        随机初始化簇中心。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入数据。
        
        返回：
            numpy.ndarray: 随机初始化的簇中心。
        """
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[random_indices]

    def _compute_distances(self, X):
        """
        计算每个样本到簇中心的距离。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入数据。
        
        返回：
            numpy.ndarray: 样本到簇中心的距离矩阵。
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, center in enumerate(self.centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return distances

    def fit(self, X):
        """
        使用训练数据对模型进行训练。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入数据。
        
        返回：
            self: 返回 KMeans 模型自身。
        """
        self.centers = self._initialize_centers(X)

        for _ in range(self.max_iters):
            # 计算每个样本到簇中心的距离，并分配到最近的簇
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)
            
            # 更新簇中心
            new_centers = np.array([X[new_labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 检查是否达到收敛条件
            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break
            
            self.centers = new_centers
            self.labels = new_labels

        return self

    def predict(self, X):
        """
        对新数据进行聚类预测。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入数据。
        
        返回：
            numpy.ndarray: 每个样本的簇标签。
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        先拟合模型，然后对数据进行聚类预测。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入数据。
        
        返回：
            numpy.ndarray: 每个样本的簇标签。
        """
        self.fit(X)
        return self.labels
