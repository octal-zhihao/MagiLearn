import numpy as np
from tqdm import tqdm

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
        # 参数检查
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("学习率 'learning_rate' 必须是正数")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("迭代次数 'n_iters' 必须是正整数")
        
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
        # 数据检查
        if not isinstance(X, np.ndarray):
            raise ValueError("输入特征 'X' 必须是 numpy.ndarray 类型")
        if not isinstance(y, np.ndarray):
            raise ValueError("目标值 'y' 必须是 numpy.ndarray 类型")
        
        # 检查 X 和 y 的形状
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(f"目标值 'y' 的样本数必须与特征矩阵 'X' 的样本数相同 ({n_samples})")

        self.weights = np.zeros(n_features)
        self.bias = 0

        # 使用 tqdm 包装迭代，显示进度条
        for _ in tqdm(range(self.n_iters), desc="Training Progress", ncols=100):
            # 预测值
            try:
                y_predicted = np.dot(X, self.weights) + self.bias
            except Exception as e:
                raise ValueError(f"在计算预测值时发生错误: {e}")

            # 计算梯度
            try:
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)
            except Exception as e:
                raise ValueError(f"在计算梯度时发生错误: {e}")

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
        # 输入检查
        if not isinstance(X, np.ndarray):
            raise ValueError("输入特征 'X' 必须是 numpy.ndarray 类型")
        
        # 确保模型已训练
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未训练，请先调用 'fit' 方法进行训练")
        
        # 预测
        try:
            return np.dot(X, self.weights) + self.bias
        except Exception as e:
            raise ValueError(f"在预测时发生错误: {e}")
