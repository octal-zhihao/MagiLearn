# 数据预处理模块 (preprocessing)

## 1.功能概览

- **数据预处理工具**：标准化、归一化、One-Hot 编码和标签编码。

以下是每个功能的使用示例。

## 2. 数据预处理工具

### 2.1 标准化 (Standardization)

**StandardScaler** 将数据标准化，使每个特征的均值为 0，方差为 1，适用于需要标准正态分布的数据。

#### 示例代码

```python
from magilearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4]])  # 示例数据

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("标准化数据:\n", X_scaled)
```

标准化数据:
 [[-1.22474487 -1.22474487]
 [ 0.          0.        ]
 [ 1.22474487  1.22474487]]

### 2.2 归一化 (Normalization) - MinMaxScaler

**MinMaxScaler** 将数据缩放到指定范围（默认 [0, 1]），适用于需要特定范围的数据。

#### 示例代码

```python
from magilearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4]])  # 示例数据

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
print("归一化数据:\n", X_normalized)
```

归一化数据:
 [[0.  0. ]
 [0.5 0.5]
 [1.  1. ]]

### 2.3 One-Hot 编码 (One-Hot Encoding)

**OneHotEncoder** 将分类特征转换为独热编码矩阵，每个类别用唯一的二进制向量表示。

#### 示例代码

```python
from magilearn.preprocessing import OneHotEncoder
import numpy as np

X = np.array([["apple"], ["banana"], ["apple"], ["orange"]])  # 示例数据

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
print("One-Hot 编码:\n", X_encoded)
```

One-Hot 编码:
 [[1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]

### 2.4 标签编码 (Label Encoding)

**LabelEncoder** 将分类标签转换为整数编码，用于将类别标签映射到整数值。

#### 示例代码

```python
from magilearn.preprocessing import LabelEncoder
import numpy as np

y = np.array(["cat", "dog", "fish", "monkey", "tiger"])  # 示例标签

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("标签编码:", y_encoded)
```

标签编码: [0 1 2 3 4]
