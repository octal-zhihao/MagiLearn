# 数据生成模块 (make_classification)

make_classification 模块提供了一个用于生成分类任务数据集的函数。该模块能够根据输入的参数生成一个具有指定特征数量、类别数以及类别之间分离性的随机数据集。它通常用于模型训练前的模拟数据生成，尤其是用来测试和验证机器学习模型。

## 参数与返回值说明
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
## 1. 数据集生成 (make_classification)
### 功能
- make_classification 函数用于生成一个用于分类任务的随机数据集。用户可以自定义特征数、类别数、信息性特征数、冗余特征数等，此外，还可以通过控制类别间隔来增强类别之间的分离性。

### 使用示例
```python
from magilearn.datasets import make_classification
import numpy as np

# 生成示例数据
X, y = make_classification(n_samples=200, n_features=10, n_informative=3, n_redundant=2, n_classes=3, class_sep=2.0, random_state=42)

print("特征矩阵 X:\n", X[:5])  # 打印前五行特征
print("标签数组 y:\n", y[:5])  # 打印前五个标签


```

## 2. 调整类别之间的分离性
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

## 3. 生成具有多个簇的数据集
### 功能
- 通过设置 n_clusters_per_class 参数，用户可以为每个类别生成多个簇。此参数在模拟多类别、复杂数据分布时特别有用。
### 使用示例
```python
from magilearn.datasets import make_classification
# 生成每个类别有多个簇的数据集
X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=2, random_state=42)

# 输出数据的前五行
print("特征矩阵 X:\n", X[:5])
print("标签数组 y:\n", y[:5])
```
## 4. 权重设置
### 功能
- weights 参数允许用户指定每个类别的样本占比。通过调整权重，您可以生成具有不平衡类分布的数据集。这在测试模型对于不平衡数据集的表现时非常有用。
### 使用示例
```python
from magilearn.datasets import make_classification
# 生成具有不平衡类别分布的数据集
weights = [0.7, 0.2, 0.1]  # 类别 0 占比 70%，类别 1 占比 20%，类别 2 占比 10%
X, y = make_classification(n_samples=200, n_features=10, n_classes=3, weights=weights, random_state=42)

# 输出数据的前五行
print("特征矩阵 X:\n", X[:5])
print("标签数组 y:\n", y[:5])

```