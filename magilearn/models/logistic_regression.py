import numpy as np

class LogisticRegression:
    """
    Logistic Regression 模型，用于二分类任务。
    
    使用梯度下降法优化模型参数，支持自定义学习率、最大迭代次数和收敛阈值。
    
    参数：
        learning_rate (float): 梯度下降的学习率，默认值为 0.01。
        num_iterations (int): 最大迭代次数，默认值为 1000。
        tol (float): 收敛阈值，当梯度变化小于此值时停止迭代，默认值为 1e-4。
    
    属性：
        coef_ (numpy.ndarray): 模型权重。
        intercept_ (float): 模型偏置。
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=1000, tol=1e-4):
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
        z = np.clip(z, -500, 500)  # 防止数值溢出
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        拟合逻辑回归模型，使用梯度下降法优化参数。
        
        参数：
            X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
            y (numpy.ndarray): 标签向量，形状为 (n_samples,)。
        """
        m, n = X.shape  # 样本数和特征数
        self.coef_ = np.zeros(n)  # 初始化权重
        self.intercept_ = 0  # 初始化偏置

        for iteration in range(self.num_iterations):
            # 计算线性模型输出
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_pred = self.sigmoid(linear_model)  # 转换为概率

            # 计算梯度
            dw = (1 / m) * np.dot(X.T, (y_pred - y))  # 权重梯度
            db = (1 / m) * np.sum(y_pred - y)  # 偏置梯度

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