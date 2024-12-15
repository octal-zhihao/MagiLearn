import numpy as np
from magilearn.datasets import load_iris
from magilearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest as SklearnSelectKBest
from sklearn.feature_selection import f_classif
# 加载 Iris 数据集
iris = load_iris()
X = iris['data']
y = iris['target']
# 选择 K 个最佳特征
selector = SelectKBest(score_func=f_classif, k=2)
# 拟合并转换数据
X_new = selector.fit_transform(X, y)
# 输出选择的特征
print("Original feature shape:", X.shape)
print("Reduced feature shape:", X_new.shape)
print("Selected features (indices):", selector.get_support(indices=True))
print("----------------------------")
# 使用 sklearn 的 SelectKBest
sklearn_selector = SklearnSelectKBest(score_func=f_classif, k=2)
X_new_sklearn = sklearn_selector.fit_transform(X, y)
print("Original feature shape (Sklearn):", X.shape)
print("Reduced feature shape (Sklearn):", X_new_sklearn.shape)
print("Selected features (indices) (Sklearn):", sklearn_selector.get_support(indices=True))