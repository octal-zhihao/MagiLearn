import numpy as np

class DecisionTreeClassifier:
    """
    决策树分类器，用于二分类任务。

    该决策树实现基于递归划分数据并使用信息增益来选择最佳特征。

    属性：
        max_depth (int): 树的最大深度。
        min_samples_split (int): 节点分裂所需的最小样本数。
        root (Node): 决策树的根节点。

    方法：
        fit(X, y): 使用训练数据构建决策树。
        predict(X): 为提供的数据预测类别标签。
        score(X, y): 计算预测的准确率。

    参数：
        max_depth (int): 树的最大深度。默认值为 10。
        min_samples_split (int): 节点分裂所需的最小样本数。默认值为 2。
    """

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        """
        决策树节点类，用于表示树中的节点。
        """
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature         # 划分特征
            self.threshold = threshold     # 划分阈值
            self.left = left               # 左子节点
            self.right = right             # 右子节点
            self.value = value             # 叶子节点的值（若为叶节点）

    def _gini(self, y):
        """
        计算 Gini 不纯度。

        参数：
            y (numpy.ndarray): 当前节点的样本标签。

        返回：
            float: Gini 不纯度。
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - sum(probabilities**2)
        return gini

    def _split(self, X, y, feature, threshold):
        """
        按照指定特征和阈值划分数据。

        参数：
            X (numpy.ndarray): 特征数据。
            y (numpy.ndarray): 标签数据。
            feature (int): 划分的特征索引。
            threshold (float): 划分阈值。

        返回：
            tuple: 分裂后的左、右子集的特征和标签数据。
        """
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    def _best_split(self, X, y):
        """
        找到最优的特征和阈值来划分数据。

        参数：
            X (numpy.ndarray): 特征数据。
            y (numpy.ndarray): 标签数据。

        返回：
            tuple: 最优特征索引、最优阈值、最小Gini不纯度。
        """
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, y_left, _, y_right = self._split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini = (len(y_left) / n_samples * self._gini(y_left)
                        + len(y_right) / n_samples * self._gini(y_right))

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth):
        """
        递归构建决策树。

        参数：
            X (numpy.ndarray): 特征数据。
            y (numpy.ndarray): 标签数据。
            depth (int): 当前节点的深度。

        返回：
            Node: 构建的决策树节点。
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)

        feature, threshold, _ = self._best_split(X, y)
        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)

        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return self.Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        """
        使用训练数据构建决策树。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。
            y (numpy.ndarray): 形状为 (n_samples,) 的目标标签。

        返回：
            None
        """
        self.root = self._build_tree(X, y, depth=0)

    def _traverse_tree(self, x, node):
        """
        递归遍历决策树来预测样本标签。

        参数：
            x (numpy.ndarray): 单个样本的特征。
            node (Node): 当前遍历的树节点。

        返回：
            int: 预测的样本标签。
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        """
        为提供的数据预测类别标签。

        参数：
            X (numpy.ndarray): 形状为 (n_samples, n_features) 的输入特征。

        返回：
            numpy.ndarray: 每个样本的预测类别标签。
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
