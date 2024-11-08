import numpy as np

class SelectFromModel:
    def __init__(self, estimator, threshold="mean"):
        """
        estimator: 训练好的模型，必须支持 coef_ 或 feature_importances_ 属性。
        threshold: 选择特征的重要性阈值，可以是：
            - "mean": 选择大于均值的特征
            - "median": 选择大于中位数的特征
            - 一个数值（例如 0.1），表示选择重要性大于该值的特征
        """
        self.estimator = estimator
        self.threshold = threshold
        self.selected_features_ = None
        
        # 自动调用模型的特征重要性计算
        self._select_features()

    def _select_features(self):
        """
        根据模型的 coef_ 或 feature_importances_ 计算特征重要性并选择特征。
        """
        # 获取模型的特征重要性或系数
        if hasattr(self.estimator, 'coef_'):
            importance = np.abs(self.estimator.coef_)
        elif hasattr(self.estimator, 'feature_importances_'):
            importance = self.estimator.feature_importances_
        else:
            raise ValueError("Estimator must have coef_ or feature_importances_ attribute")

        # 根据阈值选择特征
        if self.threshold == "mean":
            threshold_value = np.mean(importance)
        elif self.threshold == "median":
            threshold_value = np.median(importance)
        else:
            threshold_value = self.threshold
        
        # 选择重要性大于阈值的特征
        self.selected_features_ = importance >= threshold_value

    def transform(self, X):
        """
        根据计算得到的特征选择结果，返回选择后的数据。
        """
        if self.selected_features_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        # 选择重要性大于阈值的特征
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        """
        结合 _select_features 和 transform 方法
        """
        self._select_features()
        return self.transform(X)

    def get_support(self, indices=False):
        """
        获取被选择特征的布尔索引或位置索引。
        
        indices: 如果为True，返回特征索引；如果为False，返回布尔数组。
        """
        if self.selected_features_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        if indices:
            return np.where(self.selected_features_)[0]  # 返回特征的索引
        else:
            return self.selected_features_  # 返回布尔数组
