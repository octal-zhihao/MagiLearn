# encoder.py

import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = [np.unique(col) for col in X.T]

    def transform(self, X):
        transformed = []
        for i, col in enumerate(X.T):
            transformed_col = np.zeros((len(col), len(self.categories_[i])))
            for j, category in enumerate(self.categories_[i]):
                transformed_col[:, j] = (col == category).astype(float)
            transformed.append(transformed_col)
        return np.hstack(transformed)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)

    def transform(self, y):
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
