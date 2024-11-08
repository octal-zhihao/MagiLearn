import numpy as np

class LinearRegression:
    """
    线性回归模型，用于预测连续值。

    该线性回归实现使用梯度下降法进行优化。

    属性：
        learning_rate (float): 梯度下降的学习率。
        n_iters (int): 梯度下降的迭代次数。
        weights (numpy.ndarray): 模型的权重。
        bias (float): 模型的偏差项。

    方法：
        fit(X, y): 使用训练数据训练线性回归模型。
        predict(X): 为提供的数据预测连续值。

    参数：
        learning_rate (float): 优化的学习率。默认值为 0.01。
        n_iters (int): 优化的迭代次数。默认值为 1000。
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        使用训练数据训练线性回归模型。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标值。

        返回：
            None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for _ in range(self.n_iters):
            # 预测值
            y_predicted = np.dot(X, self.weights) + self.bias

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        为提供的数据预测连续值。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。

        返回：
            numpy.ndarray: 每个样本的预测值。
        """
        return np.dot(X, self.weights) + self.bias
