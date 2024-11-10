# encoder.py
# 本脚本定义了两个类，`OneHotEncoder` 和 `LabelEncoder`，用于编码类别数据。
# `OneHotEncoder` 将类别特征转换为二进制矩阵，每个类别由一个唯一的二进制向量表示。
# `LabelEncoder` 使用 0 到 n_classes-1 的值对标签进行编码，将类别标签转换为整数标签。
# 这些编码器在机器学习中被广泛使用，以便有效处理类别特征。

import numpy as np

class OneHotEncoder:
    """
    将类别特征转换为二进制矩阵，每个类别由一个唯一的二进制向量表示。
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        拟合编码器，以获得 X 中每个特征的唯一类别。

        Parameters:
        X (array-like): 用于拟合的数据，每列代表一个类别特征。
        """
        self.categories_ = [np.unique(col) for col in X.T]

    def transform(self, X):
        """
        使用拟合的类别转换输入数据 X。

        Parameters:
        X (array-like): 要转换的数据。

        Returns:
        numpy.ndarray: X 的二进制矩阵表示。
        """
        transformed = []
        for i, col in enumerate(X.T):
            transformed_col = np.zeros((len(col), len(self.categories_[i])))
            for j, category in enumerate(self.categories_[i]):
                transformed_col[:, j] = (col == category).astype(float)
            transformed.append(transformed_col)
        return np.hstack(transformed)

    def fit_transform(self, X):
        """
        拟合数据并转换数据。

        Parameters:
        X (array-like): 要拟合并转换的数据。

        Returns:
        numpy.ndarray: X 的二进制矩阵表示。
        """
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """
    使用 0 到 n_classes-1 的值对类别标签进行编码。
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """
        拟合编码器，以获得 y 中的唯一类别。

        Parameters:
        y (array-like): 用于拟合的标签。
        """
        self.classes_ = np.unique(y)

    def transform(self, y):
        """
        使用拟合的类别转换标签 y。

        Parameters:
        y (array-like): 要转换的标签。

        Returns:
        numpy.ndarray: 对应类别的整数标签数组。
        """
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y):
        """
        拟合标签并转换标签。

        Parameters:
        y (array-like): 要拟合并转换的标签。

        Returns:
        numpy.ndarray: 对应类别的整数标签数组。
        """
        self.fit(y)
        return self.transform(y)
