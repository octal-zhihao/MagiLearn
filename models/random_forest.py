import numpy as np
from collections import Counter
from decision_tree import DecisionTreeClassifier  # 假设已经实现的决策树

class RandomForestClassifier:
    """
    随机森林分类器，用于多棵决策树的集成学习。
    
    通过构建多棵决策树，并使用袋装法（bagging）来训练各决策树，最终投票确定分类结果。

    参数：
        n_estimators (int): 树的数量，默认为100。
        max_features (int): 每棵树训练时随机选择的特征数量，默认为 sqrt(n_features)。
        max_depth (int): 每棵树的最大深度，防止过拟合，默认为None。
        min_samples_split (int): 节点分裂所需的最小样本数，默认为2。
    
    方法：
        fit(X, y): 使用训练数据训练随机森林分类器。
        predict(X): 使用训练好的随机森林对数据进行预测。
    """
    
    def __init__(self, n_estimators=100, max_features=None, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        生成一个随机的样本集合，用于袋装法训练各决策树。
        
        参数：
            X (numpy.ndarray): 特征数据。
            y (numpy.ndarray): 标签数据。
            
        返回：
            (numpy.ndarray, numpy.ndarray): 袋装样本的特征和标签。
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        使用训练数据训练随机森林分类器。
        
        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标标签。
        
        返回：
            None
        """
        self.trees = []
        n_features = X.shape[1]
        max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=max_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        为给定数据预测标签。
        
        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
        
        返回：
            numpy.ndarray: 每个样本的预测标签。
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

    def score(self, X, y):
        """
        计算模型在给定数据上的准确率。
        
        参数：
            X (numpy.ndarray): 测试数据特征。
            y (numpy.ndarray): 测试数据标签。
            
        返回：
            float: 模型的准确率。
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
