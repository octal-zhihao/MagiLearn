from sklearn.metrics import accuracy_score
import numpy as np

class GridSearchCV:
    def __init__(self, estimator, param_grid, scoring=accuracy_score, cv=5):
        """
        网格搜索交叉验证

        参数：
        - estimator: 需要优化的模型，可以是一个Pipeline
        - param_grid: 超参数的网格
        - scoring: 用于评估的评分函数，默认为 accuracy_score
        - cv: 交叉验证的折数
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring  # 默认使用 accuracy_score 作为评分函数
        self.cv = cv

    def fit(self, X, y):
        """
        在给定的超参数网格上进行交叉验证

        参数：
        - X: 特征矩阵
        - y: 标签向量
        """
        best_score = -float('inf')
        best_params = None

        # 遍历参数网格
        for params in self._param_grid_generator():
            # 使用 set_params 更新Pipeline中的模型步骤的超参数
            self.estimator.set_params(**params)
            scores = []

            # 进行交叉验证
            for train_idx, test_idx in self._cross_val_split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 训练模型并进行评估
                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_test)
                scores.append(self.scoring(y_test, y_pred))  # 使用评分函数

            # 计算交叉验证的平均得分
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        # 使用最佳参数训练整个Pipeline
        self.estimator.set_params(**best_params)
        self.best_estimator_ = self.estimator  # 直接使用已经设置了最佳参数的Pipeline

        # 使用最佳参数再次训练模型
        self.best_estimator_.fit(X, y)

        return self

    def _param_grid_generator(self):
        """
        生成超参数组合的所有可能情况
        """
        import itertools
        return [dict(zip(self.param_grid.keys(), v)) for v in itertools.product(*self.param_grid.values())]

    def _cross_val_split(self, X, y):
        """
        实现简单的K折交叉验证，返回训练和测试索引
        """
        fold_size = len(X) // self.cv
        indices = np.random.permutation(len(X))
        for i in range(self.cv):
            test_idx = indices[i*fold_size : (i+1)*fold_size]
            train_idx = np.hstack([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            yield train_idx, test_idx
