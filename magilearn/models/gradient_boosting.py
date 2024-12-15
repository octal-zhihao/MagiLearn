import numpy as np
from magilearn.models.decision_tree_classifier import DecisionTreeClassifier  # 假设已实现的决策树分类器

class GradientBoostingClassifier:
    """
    梯度提升分类器，用于迭代优化和集成多个弱学习器。
    
    通过逐步增加决策树以减小预测残差，最终生成一个强分类器。

    参数：
        n_estimators (int): 基础学习器（决策树）的数量，默认为100。
        learning_rate (float): 学习率，用于缩小每棵树的贡献，默认值为0.1。
        max_depth (int): 每棵决策树的最大深度，默认值为3。
    
    方法：
        fit(X, y): 使用训练数据训练梯度提升分类器。
        predict(X): 使用训练好的模型对数据进行预测。
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        使用训练数据训练梯度提升分类器。
        
        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标标签。
        
        返回：
            None
        """
        # 初始化预测值
        y_pred = np.full(y.shape, np.mean(y))

        for _ in range(self.n_estimators):
            # 计算残差
            residual = y - y_pred
            
            # 使用残差训练一个新决策树
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, residual)
            # 更新预测值
            increment = self.learning_rate * tree.predict(X)
            y_pred += increment
            # 保存当前树
            self.trees.append(tree)

    def predict(self, X):
        """
        使用训练好的模型对数据进行预测。
        
        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
        
        返回：
            numpy.ndarray: 每个样本的预测标签。
        """
        # 初始化预测值
        y_pred = np.zeros(X.shape[0])

        # 累加所有树的预测值
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        
        # 将预测值转换为二分类标签
        return np.where(y_pred >= 0.5, 1, 0)
