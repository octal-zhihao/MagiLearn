import xgboost as xgb

class XGBoostModel:
    def __init__(self, learning_rate=0.1, max_depth=3, n_estimators=100):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
        }
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return (self.model.predict(dtest) > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
