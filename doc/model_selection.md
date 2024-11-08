# 模型选择与评估模块 (model_selection)

model_selection 模块提供了一组工具用于分割数据集、评估模型性能以及选择最佳模型参数。该模块包括以下工具，功能与 sklearn.model_selection 模块保持一致：

## 1. 数据集划分 (train_test_split.py)
### 功能
- train_test_split 函数将数据集划分为训练集和测试集，用于模型的训练和评估。

### 参数与返回值说明
```python
def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    将数据集 X 和 y 拆分为训练集和测试集。
    
    参数:
    - X: 特征矩阵
    - y: 标签向量
    - test_size: 测试集的比例（0 到 1 之间的浮动值）
    - random_state: 随机种子，保证划分的可复现性
    
    返回:
    - X_train, X_test, y_train, y_test: 划分后的训练集和测试集
    """
```

### 使用示例
```python
from magilearn.model_selection import train_test_split
import numpy as np

# 生成示例数据
X = np.arange(10).reshape((5, 2))
y = np.array([0, 1, 0, 1, 0])

# 分割数据集 (80% 训练集, 20% 测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
```

## 2. 网格搜索 (grid_search.py)
### 功能
- GridSearchCV 类用于在给定的参数网格中搜索最佳模型参数。

### 使用示例
```python
from magilearn.model_selection import GridSearchCV
from magilearn.models import LogisticRegression
import numpy as np

# 生成示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 1, 0, 1, 0])

# 定义模型和参数网格
model = LogisticRegression()
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_iter': [100, 200]
}

# 执行网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)
```

### 参数说明
```python
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
    def fit(self, X, y):
        """
        在给定的超参数网格上进行交叉验证

        参数：
        - X: 特征矩阵
        - y: 标签向量
        """
```
### 返回值说明
- best_params_: 搜索到的最佳参数组合。
- best_score_: 对应最佳参数的评分结果。

## 3. 交叉验证 (cross_val_score.py)
### 功能
- cross_val_score 函数执行 K 折交叉验证并返回每一折的得分。常用于评估模型的稳定性和泛化能力。
### 使用示例
```python
from magilearn.model_selection import cross_val_score
from magilearn.models import LogisticRegression
import numpy as np

# 生成示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 1, 0, 1, 0])

# 定义模型
model = LogisticRegression()

# 执行 3 折交叉验证
scores = cross_val_score(estimator=model, X=X, y=y, cv=3)

print("交叉验证得分:", scores)
print("平均得分:", scores.mean())
```

### 参数与返回值说明
```python
def cross_val_score(estimator, X, y, cv=5, scoring=calculate_accuracy):
    """
    执行K折交叉验证并返回每折的得分。
    
    参数:
    - estimator: 需要评估的模型
    - X: 特征矩阵
    - y: 标签向量
    - cv: 交叉验证折数
    - scoring: 用于评估的评分函数
    
    返回:
    - scores: 每个折叠的得分
    """
```