import numpy as np
from magilearn.models import DecisionTreeRegressor

class GradientBoostingClassifier:
    """
    梯度提升二分类器，用于迭代优化和集成多个弱学习器。
    
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
        # 初始化预测值为 logit(p)，其中 p 是正类的比例
        p = np.mean(y)
        y_pred = np.full(y.shape, np.log(p / (1 - p)))

        for _ in range(self.n_estimators):
            # 计算负梯度（对数损失的负梯度）
            prob_pred = 1 / (1 + np.exp(-y_pred))  # 计算概率
            residual = y - prob_pred  # 负梯度，用于拟合下一棵树
            
            # 使用残差训练一个新决策树
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            # 更新预测值
            increment = self.learning_rate * tree.predict(X)
            y_pred += increment
            # 保存当前树
            self.trees.append(tree)
            
    def predict_proba(self, X):
        """
        预测样本的概率。
        梯度提升分类器会累加所有弱学习器（决策树）的输出值，并将其通过 sigmoid 函数转换为概率值。
        
        参数:
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征矩阵。

        返回:
            numpy.ndarray: 形状为 (n_samples, 2) 的数组。
                           每一行对应 [负类概率 (P(y=0)), 正类概率 (P(y=1))]。
        """
        # 初始化预测值（弱学习器的输出累加器），初始值为0
        y_pred = np.zeros(X.shape[0])

        # 遍历所有的决策树，将每棵树的预测值（弱学习器输出）乘以学习率后累加到 y_pred
        for tree in self.trees:
            # 弱学习器的预测值累加，learning_rate 控制每棵树的贡献程度
            y_pred += self.learning_rate * tree.predict(X)
        
        # 将预测值通过 Sigmoid 转换为概率
        prob_pred = 1 / (1 + np.exp(-y_pred))
        # 返回概率数组，包含负类概率 P(y=0) 和正类概率 P(y=1)
        # P(y=0) = 1 - P(y=1)，因为二分类中概率总和为 1
        # np.vstack 作用是将两个数组按行堆叠，然后转置成形状为 (n_samples, 2)
        return np.vstack([1 - prob_pred, prob_pred]).T  # 返回 [负类概率, 正类概率]

    def predict(self, X):
        """
        预测二分类标签。
        梯度提升分类器通过概率阈值将输入样本预测为类别 0 或类别 1。
        
        参数:
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征矩阵。
        
        返回:
            numpy.ndarray: 形状为 (n_samples,) 的预测标签数组，值为 0 或 1。
        """
        # 调用 predict_proba 获取每个样本的正类概率 (P(y=1))，取 [:, 1] 即第二列
        prob_pred = self.predict_proba(X)[:, 1]

        # 使用 0.5 作为概率阈值
        # 若 P(y=1) >= 0.5，则预测标签为 1；否则，预测标签为 0
        return np.where(prob_pred >= 0.5, 1, 0)
    
    

