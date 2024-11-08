import numpy as np
class PCA:
    """
    实现主成分分析（PCA）。

    参数：
        X (numpy.ndarray): 输入数据矩阵，形状为 (n_samples, n_features)。
        n_components (int, optional): 要保留的主成分数量。默认情况下为 None，表示保留所有主成分。

    返回：
        X_pca (numpy.ndarray): 降维后的数据，形状为 (n_samples, n_components)。
        components (numpy.ndarray): 主成分矩阵，形状为 (n_components, n_features)。
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
    def fit(self, X):
        # 计算均值
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # 计算协方差矩阵
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, sorted_indices][:, :self.n_components]
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
