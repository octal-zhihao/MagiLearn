# 分类与回归模型模块 (models)

`models` 模块实现了多种机器学习模型，涵盖分类、回归和聚类任务。该模块中的模型设计类似于 `scikit-learn` 的风格，支持常用的训练、预测和评估方法，方便用户直接调用。

## 1. 逻辑回归 (logistic_regression.py)
### 功能
- `LogisticRegression` 类实现了逻辑回归模型，适用于二分类任务。
  
### 参数与方法说明
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        """
        逻辑回归模型初始化

        参数:
        - learning_rate: 学习率
        - max_iter: 最大迭代次数
        """
    def fit(self, X, y):
        """
        训练模型

        参数:
        - X: 特征矩阵
        - y: 标签向量
        """
    def predict(self, X):
        """
        预测标签

        参数:
        - X: 特征矩阵
        """
    def score(self, X, y):
        """
        计算模型的准确率

        参数:
        - X: 特征矩阵
        - y: 标签向量
        """
```

### 使用示例
```python
from magilearn.models import LogisticRegression

model = LogisticRegression(learning_rate=0.1, max_iter=500)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

## 2. 线性回归 (linear_regression.py)
### 功能
- `LinearRegression` 类实现了线性回归模型，用于回归任务，拟合输入特征与连续标签之间的线性关系。

### 参数与方法说明
```python
class LinearRegression:
    def __init__(self):
        """
        线性回归模型初始化
        """
    def fit(self, X, y):
        """
        训练模型

        参数:
        - X: 特征矩阵
        - y: 标签向量
        """
    def predict(self, X):
        """
        预测标签

        参数:
        - X: 特征矩阵
        """
```
### 使用示例
```python
from magilearn.models import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 3. 决策树分类 (decision_tree.py)
### 功能
- `DecisionTreeClassifier` 实现了决策树分类器，适用于分类任务，能够处理连续和离散数据。
### 参数与方法说明
```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        决策树分类器初始化

        参数:
        - max_depth: 树的最大深度
        """
    def fit(self, X, y):
        """
        训练模型
        """
    def predict(self, X):
        """
        预测标签
        """
```
### 使用示例
```python
from magilearn.models import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
```

## 4. 随机森林分类器 (random_forest.py)
### 功能
- `RandomForestClassifier` 实现了随机森林分类器，通过集成多个决策树提高模型的准确性和泛化能力。
### 参数与方法说明
```python
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        """
        随机森林分类器初始化

        参数:
        - n_estimators: 树的数量
        - max_depth: 每棵树的最大深度
        """
    def fit(self, X, y):
        """
        训练模型
        """
    def predict(self, X):
        """
        预测标签
        """
```
### 使用示例
```python
from magilearn.models import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

## 5. 梯度提升分类器 (gradient_boosting.py)
### 功能
- `GradientBoostingClassifier` 实现了梯度提升分类器，使用提升方法通过逐步构建多个弱分类器来提高预测精度。
### 参数与方法说明
```python
class GradientBoostingClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        """
        梯度提升分类器初始化

        参数:
        - n_estimators: 弱分类器数量
        - learning_rate: 学习率
        """
    def fit(self, X, y):
        """
        训练模型
        """
    def predict(self, X):
        """
        预测标签
        """
```
### 使用示例
```python
from magilearn.models import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=50)
model.fit(X_train, y_train)
```

## 6. K均值聚类 (k_means.py)
### 功能
- `KMeans` 实现了 K 均值聚类算法，用于无监督聚类，将样本分配到 k 个簇中。
### 参数与方法说明
```python
class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        """
        K 均值聚类模型初始化

        参数:
        - n_clusters: 簇的数量
        - max_iter: 最大迭代次数
        """
    def fit(self, X):
        """
        训练模型
        """
    def predict(self, X):
        """
        预测簇标签
        """
```
### 使用示例
```python
from magilearn.models import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
```