import numpy as np

class LogisticRegression:
    """
    逻辑回归模型，用于二分类任务。

    该逻辑回归实现使用梯度下降法进行优化。

    属性：
        learning_rate (float): 梯度下降的学习率。
        n_iters (int): 梯度下降的迭代次数。
        weights (numpy.ndarray): 模型的权重。
        bias (float): 模型的偏差项。

    方法：
        fit(X, y): 使用训练数据训练逻辑回归模型。
        predict(X): 为提供的数据预测类别标签。

    参数：
        learning_rate (float): 优化的学习率。默认值为0.01。
        n_iters (int): 优化的迭代次数。默认值为1000。
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        """
        应用 sigmoid 函数。

        参数：
            x (numpy.ndarray): 输入值。

        返回：
            numpy.ndarray: 输入的 sigmoid 值。
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        使用训练数据训练逻辑回归模型。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标标签。

        返回：
            None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        为提供的数据预测类别标签。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。

        返回：
            list: 每个样本的预测类别标签（0或1）。
        """
        model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(model)
        return [1 if i > 0.5 else 0 for i in y_pred]
