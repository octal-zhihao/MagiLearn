# encoder.py
# This script defines two classes, `OneHotEncoder` and `LabelEncoder`, for encoding categorical data.
# `OneHotEncoder` transforms categorical features into a binary matrix, with each category represented by a unique binary vector.
# `LabelEncoder` encodes labels with a value between 0 and n_classes-1, converting categorical labels to integer labels.
# These encoders are commonly used in machine learning to handle categorical features effectively.

import numpy as np

class OneHotEncoder:
    """
    Transforms categorical features into a binary matrix, with each category represented by a unique binary vector.
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        Fit the encoder to the unique categories for each feature in X.

        Parameters:
        X (array-like): Data to be used for fitting, where each column represents a categorical feature.
        """
        self.categories_ = [np.unique(col) for col in X.T]

    def transform(self, X):
        """
        Transform the input data X using the fitted categories.

        Parameters:
        X (array-like): Data to be transformed.

        Returns:
        numpy.ndarray: Transformed binary matrix representation of X.
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
        Fit to data, then transform it.

        Parameters:
        X (array-like): Data to be fitted and transformed.

        Returns:
        numpy.ndarray: Transformed binary matrix representation of X.
        """
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """
    Encodes categorical labels with values between 0 and n_classes-1.
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """
        Fit the encoder to the unique classes in y.

        Parameters:
        y (array-like): Labels to be used for fitting.
        """
        self.classes_ = np.unique(y)

    def transform(self, y):
        """
        Transform the labels y using the fitted classes.

        Parameters:
        y (array-like): Labels to be transformed.

        Returns:
        numpy.ndarray: Array of integer labels corresponding to the classes.
        """
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y):
        """
        Fit to labels, then transform them.

        Parameters:
        y (array-like): Labels to be fitted and transformed.

        Returns:
        numpy.ndarray: Array of integer labels corresponding to the classes.
        """
        self.fit(y)
        return self.transform(y)
