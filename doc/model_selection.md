# 模型选择与评估模块 (model_selection)

model_selection 模块提供了一组工具用于分割数据集、评估模型性能以及选择最佳模型参数。该模块包括以下工具：

## 1. 网格搜索 (grid_search.py)
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

## 2. 交叉验证 (cross_val_score.py)
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

## 3. 模型保存与载入模块 (save_model.py & load_model.py)

### 参数与返回值说明
```python
def save_model(model, filename):
    """
    保存训练好的模型到文件。

    参数：
        model : 训练好的模型对象
        filename : str
            模型保存的文件路径和名称
    """
```