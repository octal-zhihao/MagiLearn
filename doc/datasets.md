# 数据集模块

## 1. 加载数据集 (load.iris.py)
### 功能
- 加载并返回鸢尾花数据集（与 sklearn 一致的结构），数据从 `datasets/data/` 路径中的多个文件中读取。

### 参数与返回值说明
```python
def load_iris(return_X_y=False):
    """
    加载并返回鸢尾花数据集（与 sklearn 一致的结构），数据从 `datasets/data/` 路径中的多个文件中读取。
    
    参数:
    - return_X_y (bool): 如果为 True，返回 (X, y) 而不是字典格式。
    
    返回值:
    - 如果 return_X_y=False，返回字典格式，包含 'data', 'target', 'target_names', 'feature_names', 'DESCR'。
    - 如果 return_X_y=True，返回 (X, y) 元组。
    """
```
### 使用示例
```python
from magilearn.datasets.load_iris import load_iris

# 加载数据集
iris = load_iris()

# 查看数据集信息
print("特征名称:", iris['feature_names'])
print("数据形状:", iris['data'].shape)
print("目标名称:", iris['target_names'])
print("描述信息:", iris['DESCR'])
print("数据示例:")
print(iris['data'][:5])  # 打印前5个样本
print("目标示例:")
print(iris['target'][:5])  # 打印前5个标签
```

## 2. 数据集划分 (train_test_split.py)
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

## 3. 数据集生成 (make_classification)
### 功能
- make_classification 函数用于生成一个用于分类任务的随机数据集。用户可以自定义特征数、类别数、信息性特征数、冗余特征数等，此外，还可以通过控制类别间隔来增强类别之间的分离性。

### 参数与返回值说明
```python
def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                        n_clusters_per_class=1, n_classes=2, weights=None, random_state=None, class_sep=1.0):
    """
    生成一个用于分类的随机数据集，增强类别之间的分离性。

    参数：
    - n_samples: 样本数量（默认为100）
    - n_features: 特征总数（默认为20）
    - n_informative: 信息性特征数量（默认为2）
    - n_redundant: 冗余特征数量（默认为2）
    - n_clusters_per_class: 每个类别的聚类数量（默认为1）
    - n_classes: 数据集中的类别数量（默认为2）
    - weights: 每个类别的样本比例（默认为None，均匀分布）
    - random_state: 随机种子，保证划分的可复现性（默认为None）
    - class_sep: 类别间隔的大小，用于控制类别中心的分离程度（默认为1.0）

    返回：
    - X: 特征矩阵，形状为 (n_samples, n_features)
    - y: 标签数组，形状为 (n_samples,)
    """

```

### 使用示例
```python
from magilearn.datasets import make_classification
import numpy as np

# 生成示例数据
X, y = make_classification(n_samples=200, n_features=10, n_informative=3, n_redundant=2, n_classes=3, class_sep=2.0, random_state=42)

print("特征矩阵 X:\n", X[:5])  # 打印前五行特征
print("标签数组 y:\n", y[:5])  # 打印前五个标签


```

## 4. 调整类别之间的分离性
### 功能
- class_sep 参数能够控制不同类别之间的分离度。默认情况下，类别之间的间隔为 1.0。增大 class_sep 的值会增加类别之间的分离度，生成的样本之间的区分更加明显。
### 影响
- 增加 class_sep 值：会使类别之间的分布更加分开，可能更容易实现准确的分类。 
- 减小 class_sep 值：类别之间的重叠增加，生成的分类问题变得更加困难。
### 使用示例
```python
from magilearn.datasets import make_classification
# 生成具有较大类别分离的数据集
X, y = make_classification(n_samples=200, n_features=10, n_classes=3, class_sep=3.0, random_state=42)

# 输出数据的前五行
print("特征矩阵 X:\n", X[:5])
print("标签数组 y:\n", y[:5])

```
