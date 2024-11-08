import numpy as np

class LogisticRegression:
    """
    逻辑回归模型，支持二分类和多类分类（通过 OvR 策略）。

    参数：
        learning_rate : float, default=0.01
            梯度下降的学习率。
        n_iters : int, default=1000
            梯度下降的最大迭代次数。
        regularization : str, default=None
            正则化方式，可选 'l2'。若为 None，则不使用正则化。
        C : float, default=1.0
            正则化强度的倒数。较小的值表示更强的正则化。
        multi_class : str, default='ovr'
            用于多类分类任务的策略。当前支持 'ovr' (one-vs-rest) 策略。
    """

    def __init__(self, learning_rate=0.01, n_iters=1000, regularization=None, C=1.0, multi_class='ovr'):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.C = C
        self.multi_class = multi_class
        self.weights = None
        self.bias = None
        self.classes_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        """
        训练逻辑回归模型。

        参数：
            X : numpy.ndarray, shape (n_samples, n_features)
                输入特征矩阵。
            y : numpy.ndarray, shape (n_samples,)
                目标标签。支持二分类和多类分类任务。
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        if len(self.classes_) > 2:
            # 多类分类任务 - One-vs-Rest
            self.models_ = []
            for cls in self.classes_:
                y_binary = np.where(y == cls, 1, 0)
                model = LogisticRegression(
                    learning_rate=self.learning_rate,
                    n_iters=self.n_iters,
                    regularization=self.regularization,
                    C=self.C
                )
                model._fit_binary(X, y_binary)
                self.models_.append(model)
        else:
            # 二分类任务
            self._fit_binary(X, y)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        # 梯度下降
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # L2 正则化
            if self.regularization == 'l2':
                dw += (1 / self.C) * self.weights / n_samples

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        为输入样本预测类别标签。

        参数：
            X : numpy.ndarray, shape (n_samples, n_features)
                输入特征矩阵。

        返回：
            y_pred : list
                每个样本的预测类别标签。
        """
        if len(self.classes_) > 2:
            # 多类分类任务
            probs = np.array([model.predict_proba(X)[:, 1] for model in self.models_]).T
            return np.array([self.classes_[np.argmax(prob)] for prob in probs])
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def predict_proba(self, X):
        """
        返回每个输入样本属于每个类别的概率估计。

        参数：
            X : numpy.ndarray, shape (n_samples, n_features)
            输入特征矩阵。

        返回：
            y_proba : numpy.ndarray
                每个样本的类别概率。
        """
        if len(self.classes_) > 2:
            # 多类分类任务
            probs = np.array([model.predict_proba(X)[:, 1] for model in self.models_]).T
            return probs / probs.sum(axis=1, keepdims=True)
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            return np.vstack([1 - y_pred, y_pred]).T
