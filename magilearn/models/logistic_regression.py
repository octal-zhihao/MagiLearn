import numpy as np
from tqdm import tqdm

class LogisticRegression:
    """
    Logistic Regression 模型，用于二分类任务。
    
    使用梯度下降法优化模型参数，支持自定义学习率、最大迭代次数和收敛阈值。
    
    参数：
        learning_rate (float): 梯度下降的学习率，默认值为 0.01。
        num_iterations (int): 最大迭代次数，默认值为 5000。
        tol (float): 收敛阈值，当梯度变化小于此值时停止迭代，默认值为 1e-4。
    
    属性：
        coef_ (numpy.ndarray): 模型权重。
        intercept_ (float): 模型偏置。
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=5000, tol=1e-4):
        # 检查输入参数
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("学习率 'learning_rate' 必须是正数")
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("最大迭代次数 'num_iterations' 必须是正整数")
        if not isinstance(tol, (int, float)) or tol < 0:
            raise ValueError("收敛阈值 'tol' 必须是非负数")
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid 激活函数，将线性输出转换为概率值。
        
        参数：
            z (numpy.ndarray): 线性模型的输出。
        
        返回：
            numpy.ndarray: Sigmoid 转换后的概率值。
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        拟合逻辑回归模型，使用梯度下降法优化参数。
        
        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
            y (numpy.ndarray): 标签向量，形状为 (n_samples,)。
        """
        # 数据检查
        if not isinstance(X, np.ndarray):
            raise ValueError("输入特征矩阵 'X' 必须是 numpy.ndarray 类型")
        if not isinstance(y, np.ndarray):
            raise ValueError("标签 'y' 必须是 numpy.ndarray 类型")
        
        m, n = X.shape  # 样本数和特征数
        
        if y.shape[0] != m:
            raise ValueError(f"标签 'y' 的样本数必须与特征矩阵 'X' 的样本数相同 ({m})")

        # 初始化权重和偏置
        self.coef_ = np.zeros(n)  # 初始化权重
        self.intercept_ = 0  # 初始化偏置

        for iteration in tqdm(range(self.num_iterations), desc="Training Logistic Regression", unit="iter"):
            # 计算线性模型输出
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_pred = self.sigmoid(linear_model)  # 转换为概率

            # 计算梯度
            dw = (1 / m) * np.dot(X.T, (y_pred - y))  # 权重梯度
            db = (1 / m) * np.sum(y_pred - y)  # 偏置梯度

            # 防止梯度更新过大导致数值不稳定
            if np.isnan(dw).any() or np.isnan(db):
                raise ValueError("在计算梯度时出现了无效值（NaN），可能是数值不稳定")
            
            # 参数更新
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            # 检查是否收敛
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

    def predict_proba(self, X):
        """
        预测每个样本属于正类的概率。
        
        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
        
        返回：
            numpy.ndarray: 样本属于正类的概率，形状为 (n_samples,)。
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("输入特征矩阵 'X' 必须是 numpy.ndarray 类型")
        
        linear_model = np.dot(X, self.coef_) + self.intercept_
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        对样本进行分类预测。
        
        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
        
        返回：
            numpy.ndarray: 二分类预测结果（0 或 1），形状为 (n_samples,)。
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)  # 使用 0.5 作为分类阈值
