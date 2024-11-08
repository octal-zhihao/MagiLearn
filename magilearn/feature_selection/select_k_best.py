import numpy as np
class SelectKBest:
    """
    选择 K 个最佳特征的类。

    参数：
        score_func (callable): 计算特征评分的函数，接受 (X, y) 作为输入并返回特征分数。
        k (int): 要选择的最佳特征的数量。
    """
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.support_ = None
        self.selected_indices_ = None  # 添加这个属性来保存选择的特征索引
    def fit(self, X, y):
        # 计算特征评分
        self.scores_ = self.score_func(X, y)[0]  # 仅取得分部分
        # 确保 scores_ 是一维数组
        if self.scores_.ndim != 1:
            raise ValueError("The score function must return a one-dimensional array of scores.")
        # 选择 K 个最佳特征的索引
        if self.k > 0:
            indices = np.argsort(self.scores_)[-self.k:]  # 选择得分最高的特征索引
        else:
            indices = np.arange(X.shape[1])  # 如果 k <= 0，选择所有特征

        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[indices] = True  # 标记选择的特征
        self.selected_indices_ = indices  # 保存选择的特征索引
    def transform(self, X):
        # 根据选择的特征索引进行转换，确保返回的是二维数组
        return X[:, self.selected_indices_]
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    def get_support(self, indices=False):
        """
        返回被选特征的支持布尔数组或索引。

        参数：
            indices (bool): 如果为 True，则返回索引，否则返回布尔数组。

        返回：
            np.ndarray: 特征的支持数组或索引。
        """
        if indices:
            return self.selected_indices_  # 返回索引数组
        return self.support_  # 返回布尔数组




