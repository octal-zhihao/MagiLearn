# scalers.py
# This script defines two classes, `StandardScaler` and `MinMaxScaler`, for data preprocessing.
# `StandardScaler` standardizes features by removing the mean and scaling to unit variance.
# `MinMaxScaler` scales features to a specified range, typically [0, 1].
# These scalers are commonly used in machine learning to normalize data,
# improving model performance and training stability.

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation for each feature in X.

        Parameters:
        X (array-like): The data to calculate mean and std on.
        y (None): Ignored, exists for compatibility with other preprocessing classes.
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X):
        """
        Standardize the features of X using the computed mean and std.

        Parameters:
        X (array-like): The data to be standardized.

        Returns:
        array-like: Standardized data.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The data to be standardized.
        y (None): Ignored, exists for compatibility with other preprocessing classes.

        Returns:
        array-like: Standardized data.
        """
        self.fit(X, y)
        return self.transform(X)


class MinMaxScaler:
    """
    Scales features to a specified range, defaulting to [0, 1].
    """

    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum for each feature in X.

        Parameters:
        X (array-like): The data to calculate min and max on.
        y (None): Ignored, exists for compatibility with other preprocessing classes.
        """
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X):
        """
        Scale the features of X to the specified range.

        Parameters:
        X (array-like): The data to be scaled.

        Returns:
        array-like: Scaled data.
        """
        X_std = (X - self.min_) / (self.max_ - self.min_)
        return X_std * (self.range_max - self.range_min) + self.range_min

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The data to be scaled.
        y (None): Ignored, exists for compatibility with other preprocessing classes.

        Returns:
        array-like: Scaled data.
        """
        self.fit(X, y)
        return self.transform(X)
