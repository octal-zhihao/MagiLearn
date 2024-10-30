from sklearn.metrics import accuracy_score

class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self.steps_dict = {name: step for name, step in steps}

    def fit(self, X, y):
        for name, step in self.steps:
            X = step.fit_transform(X, y) if hasattr(step, 'fit_transform') else step.fit(X, y)
        return self

    def predict(self, X):
        for name, step in self.steps:
            X = step.transform(X) if hasattr(step, 'transform') else step.predict(X)
        return X

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))  # 需要导入accuracy_score
