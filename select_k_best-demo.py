import numpy as np
from sklearn.datasets import load_iris
from feature_selection import SelectKBest
# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target
# 选择 K 个最佳特征
selector = SelectKBest(score_func=f_classif, k=2)
# 拟合并转换数据
X_new = selector.fit_transform(X, y)
# 输出选择的特征
print("Original feature shape:", X.shape)
print("Reduced feature shape:", X_new.shape)
print("Selected features (indices):", selector.get_support(indices=True))