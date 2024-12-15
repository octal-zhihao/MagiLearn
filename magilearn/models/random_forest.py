import numpy as np
from collections import Counter
from magilearn.models.decision_tree import DecisionTreeClassifier


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
        # 参数检查
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators 必须是正整数")
        if max_features not in [None, 'sqrt'] and (not isinstance(max_features, int) or max_features <= 0):
            raise ValueError("max_features 必须是 None, 'sqrt' 或正整数")
        if not isinstance(min_samples_split, int) or min_samples_split <= 0:
            raise ValueError("min_samples_split 必须是正整数")

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
        # 数据检查
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X 和 y 必须是 numpy.ndarray 类型")
        if X.shape[0] != y.shape[0]:
            raise ValueError("特征矩阵 X 和标签向量 y 的样本数必须一致")

        self.trees = []
        n_features = X.shape[1]

        # 处理 max_features 参数，如果是 'sqrt' 则计算平方根
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = self.max_features or n_features  # 默认使用所有特征

        # 使用 tqdm 包装训练过程
        for i in tqdm(range(self.n_estimators), desc="Training Trees", ncols=100):
            try:
                tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                              max_features=max_features)
                X_sample, y_sample = self._bootstrap_sample(X, y)
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)
            except Exception as e:
                print(f"训练第 {i + 1} 棵树时发生错误: {e}")
                continue

    def predict(self, X):
        """
        为给定数据预测标签。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。

        返回：
            numpy.ndarray: 每个样本的预测标签。
        """
        # 输入检查
        if not isinstance(X, np.ndarray):
            raise ValueError("输入特征 X 必须是 numpy.ndarray 类型")

        # 获取每棵树的预测结果
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        try:
            return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
        except Exception as e:
            raise ValueError(f"在预测时发生错误: {e}")
