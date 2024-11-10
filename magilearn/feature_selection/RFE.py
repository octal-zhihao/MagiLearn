import numpy as np

class RFE:
    """
    递归特征消除 (RFE) 实现，用于特征选择。

    参数：
        estimator (object): 用于评估特征权重的模型，必须实现 fit 和 coef_ 或 feature_importances_ 属性。
        n_features_to_select (int): 要选择的特征数量。
    """
    def __init__(self, estimator, n_features_to_select=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.support_ = None
        self.ranking_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.support_ = np.ones(n_features, dtype=bool)  # 初始化所有特征为被选中状态
        self.ranking_ = np.ones(n_features, dtype=int) * -1  # 初始化排名为-1以区分未处理的特征

        current_features = n_features  # 当前特征数量
        ranking_order = 1              # 排名从1开始

        while current_features > self.n_features_to_select:
            # 使用当前选择的特征训练模型
            self.estimator.fit(X[:, self.support_], y)

            # 获取特征权重，根据模型不同使用不同的属性
            if hasattr(self.estimator, 'coef_'):
                importances = np.abs(self.estimator.coef_).flatten()
            elif hasattr(self.estimator, 'feature_importances_'):
                importances = np.abs(self.estimator.feature_importances_)
            else:
                raise ValueError("Estimator does not have 'coef_' or 'feature_importances_' attribute")
            # print(f"Current importances: {importances}")

            # 找到当前最不重要的特征
            selected_feature_indices = np.where(self.support_)[0]
            least_important_index = np.argmin(importances[selected_feature_indices])  # 找到最不重要特征的索引
            # print(f"Least important feature: {selected_feature_indices[least_important_index]}")  # 输出被删除的特征

            # 确保索引不越界
            if least_important_index >= len(selected_feature_indices):
                least_important_index = len(selected_feature_indices) - 1

            feature_to_remove = selected_feature_indices[least_important_index]

            # 更新支持数组和排名
            self.support_[feature_to_remove] = False
            self.ranking_[feature_to_remove] = ranking_order
            ranking_order += 1

            current_features -= 1

        # 将剩余选择的特征排名设为1
        self.ranking_[self.support_] = 1

    def transform(self, X):
        # 根据支持数组返回选定的特征
        return X[:, self.support_]

    def fit_transform(self, X, y):
        # 同时进行拟合和转换
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        # 返回支持数组，指定 indices=True 返回特征索引
        if indices:
            return np.where(self.support_)[0]
        return self.support_



