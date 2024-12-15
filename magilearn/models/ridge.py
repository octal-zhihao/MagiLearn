import numpy as np
from tqdm import tqdm

class Ridge:
    """
    岭回归（Ridge Regression），是一种带有 L2 正则化项的线性回归模型。
    通过最小化损失函数，并在损失函数中加入 L2 正则化项，来避免过拟合。

    参数：
        alpha (float): 正则化强度，控制模型的复杂度，默认值为 1.0。
        learning_rate (float): 梯度下降法的学习率，默认值为 0.01。
        n_iters (int): 最大迭代次数，默认值为 10000。
        tol (float): 收敛阈值，当梯度变化小于此值时停止迭代，默认值为 1e-4。

    属性：
        coef_ (numpy.ndarray): 模型权重。
        intercept_ (float): 模型偏置。
    """
    
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iters=10000, tol=1e-4):
        self.alpha = alpha  # 正则化强度
        self.learning_rate = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.tol = tol  # 收敛阈值
        self.coef_ = None  # 模型权重
        self.intercept_ = None  # 偏置项
    
    def fit(self, X, y):
        """
        拟合模型，使用带有 L2 正则化的梯度下降法。

        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
            y (numpy.ndarray): 目标值，形状为 (n_samples,)。

        返回：
            None
        """
        try:
            m, n = X.shape  # 样本数和特征数
            self.coef_ = np.zeros(n)  # 初始化权重
            self.intercept_ = 0  # 初始化偏置

            # 使用tqdm来显示训练进度条
            for _ in tqdm(range(self.n_iters), desc="训练进度", ncols=100, leave=False):
                # 预测值
                y_pred = np.dot(X, self.coef_) + self.intercept_

                # 计算梯度
                dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.alpha / m) * self.coef_  # L2 正则化项
                db = (1 / m) * np.sum(y_pred - y)

                # 更新权重和偏置
                self.coef_ -= self.learning_rate * dw
                self.intercept_ -= self.learning_rate * db

                # 检查是否收敛
                if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                    print(f"模型已收敛，迭代次数: {_ + 1}")
                    break
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
    
    def predict(self, X):
        """
        使用训练好的模型对数据进行预测。

        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。

        返回：
            numpy.ndarray: 预测值，形状为 (n_samples,)。
        """
        try:
            return np.dot(X, self.coef_) + self.intercept_
        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            return None
