import numpy as np
from sklearn.metrics import accuracy_score

def cross_val_score(estimator, X, y, cv=5, scoring=accuracy_score):
    """
    执行K折交叉验证并返回每折的得分。
    
    Parameters:
    - estimator: 需要评估的模型
    - X: 特征矩阵
    - y: 标签向量
    - cv: 交叉验证折数
    - scoring: 用于评估的评分函数
    
    Returns:
    - scores: 每个折叠的得分
    """
    scores = []

    # 切分数据并进行交叉验证
    fold_size = len(X) // cv
    indices = np.random.permutation(len(X))

    for i in range(cv):
        test_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.hstack([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练模型并进行预测
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        # 计算并存储得分
        scores.append(scoring(y_test, y_pred))

    return np.array(scores)
