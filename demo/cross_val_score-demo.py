import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# from magilearn.model_selection import cross_val_score

# 生成一个随机的二分类数据集
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# 创建一个逻辑回归模型
model = LogisticRegression()

# 使用 cross_val_score 进行交叉验证
scores = cross_val_score(estimator=model, X=X, y=y, cv=5)

# 输出交叉验证的得分
print("Cross-validation scores:", scores)
print("Mean score:", np.mean(scores))
