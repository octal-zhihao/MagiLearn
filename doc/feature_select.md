# 特征选择模块 (feature_selection)

feature_selection 模块提供了多种特征选择的方法，用于选择对模型效果最重要的特征。该模块包括降维、递归特征消除、基于模型的特征选择以及选择最优的 K 个特征等多种方法，功能与 sklearn.feature_selection 模块相似。
## 1. 主成分分析 (PCA)
### 功能
- PCA 类实现了主成分分析，用于降维处理，将高维特征映射到低维空间，同时保留数据的最大方差。它通过构造特征的线性组合来减少冗余信息，适合于数据降维和噪声过滤。
### 参数与返回值说明
```python
class PCA:
    """
    实现主成分分析（PCA）。

    参数：
        X (numpy.ndarray): 输入数据矩阵，形状为 (n_samples, n_features)。
        n_components (int, optional): 要保留的主成分数量。默认情况下为 None，表示保留所有主成分。

    返回：
        X_pca (numpy.ndarray): 降维后的数据，形状为 (n_samples, n_components)。
        components (numpy.ndarray): 主成分矩阵，形状为 (n_components, n_features)。
    """
```

### 使用示例
```python
from magilearn.feature_selection import PCA
import numpy as np

# 生成示例数据
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

# PCA 降维
pca = PCA(n_components=1)
pca.fit(X)
X_transformed = pca.transform(X)

print("降维后的数据:\n", X_transformed)


```

## 2. 递归特征消除 (RFE)
### 功能
- RFE 类实现了递归特征消除算法，通过递归训练模型、评估特征重要性，逐步移除不重要的特征，直到达到预期的特征数量。
### 使用示例
```python
from magilearn.feature_selection import RFE
from magilearn.models import LogisticRegression
import numpy as np

# 生成示例数据
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])

# 使用 RFE 选择最优特征
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2)
rfe.fit(X, y)
X_transformed = rfe.transform(X)

print("选择的特征:\n", X_transformed)

```

### 参数说明
```python
class RFE:
    """
    递归特征消除 (RFE) 实现，用于特征选择。

    参数：
        estimator (object): 用于评估特征权重的模型，必须实现 fit 和 coef_ 或 feature_importances_ 属性。
        n_features_to_select (int): 要选择的特征数量。
    """
```

## 3. 基于模型的特征选择 (SelectFromModel)
### 功能
- SelectFromModel 类利用带有特征权重的模型（如线性回归、决策树）进行特征选择，选择权重值高于指定阈值的特征。
### 使用示例
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
# 去掉sklearn.后的SelectFromModel效果一致
from magilearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

selector = SelectFromModel(model, threshold="mean")  # 选择重要性大于平均值的特征
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("Selected features:", selector.get_support(indices=True))

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on original features: {accuracy:.4f}")

model_selected = DecisionTreeClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)

y_pred = model_selected.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on selected features: {accuracy:.4f}")

```

### 参数与返回值说明
```python
    def __init__(self, estimator, threshold="mean"):
        """
        estimator: 训练好的模型，必须支持 coef_ 或 feature_importances_ 属性。
        threshold: 选择特征的重要性阈值，可以是：
            - "mean": 选择大于均值的特征
            - "median": 选择大于中位数的特征
            - 一个数值（例如 0.1），表示选择重要性大于该值的特征
        """

```

## 4. 选择 k 个最佳特征 (SelectKBest)
### 功能
- SelectKBest 函数用于选择最优的 K 个特征，通常根据单变量统计指标（如卡方、F 值）进行选择。
### 使用示例
```python
import numpy as np
from sklearn.datasets import load_iris
from magilearn.feature_selection import SelectKBest
from sklearn.feature_selection import 

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

```
### 参数与返回值说明
```python
class SelectKBest:
    """
    选择 K 个最佳特征的类。

    参数：
        score_func (callable): 计算特征评分的函数，接受 (X, y) 作为输入并返回特征分数。
        k (int): 要选择的最佳特征的数量。
        indices (bool): 如果为 True，则返回索引，否则返回布尔数组。
    """

```