import numpy as np

class SVMClassifier:
    """
    支持向量机（SVM）分类器，用于二分类任务。

    该 SVM 实现使用梯度下降法优化 Hinge 损失函数。

    属性：
        learning_rate (float): 梯度下降的学习率。
        regularization_strength (float): 正则化强度系数。
        n_iters (int): 梯度下降的迭代次数。
        weights (numpy.ndarray): 模型的权重。
        bias (float): 模型的偏差项。

    方法：
        fit(X, y): 使用训练数据训练 SVM 模型。
        predict(X): 为提供的数据预测类别标签。

    参数：
        learning_rate (float): 优化的学习率。默认值为 0.001。
        regularization_strength (float): 正则化强度系数。默认值为 0.01。
        n_iters (int): 优化的迭代次数。默认值为 1000。
    """

    def __init__(self, learning_rate=0.001, regularization_strength=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        使用训练数据训练 SVM 模型。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标标签（每个标签应为 +1 或 -1）。

        返回：
            None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    dw = 2 * self.regularization_strength * self.weights
                    db = 0
                else:
                    dw = 2 * self.regularization_strength * self.weights - np.dot(x_i, y[idx])
                    db = -y[idx]

                # 更新权重和偏差项
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        为提供的数据预测类别标签。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。

        返回：
            numpy.ndarray: 每个样本的预测类别标签（+1 或 -1）。
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
