# scalers.py
# 本脚本定义了两个类，`StandardScaler` 和 `MinMaxScaler`，用于数据预处理。
# `StandardScaler` 通过去除均值并缩放到单位方差来标准化特征。
# `MinMaxScaler` 将特征缩放到指定的范围，通常是 [0, 1]。
# 这些缩放器在机器学习中常用于归一化数据，
# 以提升模型性能和训练稳定性。

class StandardScaler:
    """
    通过去除均值并缩放到单位方差来标准化特征。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        计算 X 中每个特征的均值和标准差。

        Parameters:
        X (array-like): 用于计算均值和标准差的数据。
        y (None): 忽略此参数，仅为兼容其他预处理类。
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X):
        """
        使用计算的均值和标准差对 X 的特征进行标准化。

        Parameters:
        X (array-like): 要标准化的数据。

        Returns:
        array-like: 标准化后的数据。
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        """
        拟合数据，然后对其进行标准化。

        Parameters:
        X (array-like): 要标准化的数据。
        y (None): 忽略此参数，仅为兼容其他预处理类。

        Returns:
        array-like: 标准化后的数据。
        """
        self.fit(X, y)
        return self.transform(X)


class MinMaxScaler:
    """
    将特征缩放到指定范围，默认为 [0, 1]。
    """

    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, X, y=None):
        """
        计算 X 中每个特征的最小值和最大值。

        Parameters:
        X (array-like): 用于计算最小值和最大值的数据。
        y (None): 忽略此参数，仅为兼容其他预处理类。
        """
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X):
        """
        将 X 的特征缩放到指定范围。

        Parameters:
        X (array-like): 要缩放的数据。

        Returns:
        array-like: 缩放后的数据。
        """
        X_std = (X - self.min_) / (self.max_ - self.min_)
        return X_std * (self.range_max - self.range_min) + self.range_min

    def fit_transform(self, X, y=None):
        """
        拟合数据，然后对其进行缩放。

        Parameters:
        X (array-like): 要缩放的数据。
        y (None): 忽略此参数，仅为兼容其他预处理类。

        Returns:
        array-like: 缩放后的数据。
        """
        self.fit(X, y)
        return self.transform(X)
