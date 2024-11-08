class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, X, y=None):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X):
        X_std = (X - self.min_) / (self.max_ - self.min_)
        return X_std * (self.range_max - self.range_min) + self.range_min

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
