import numpy as np
from magilearn.preprocessing import OneHotEncoder, LabelEncoder

# 原始分类数据
X = np.array([['红色'], ['蓝色'], ['绿色'], ['红色']])

# 使用 OneHotEncoder
encoder1 = OneHotEncoder()
X_encoded = encoder1.fit_transform(X)
print("独热编码后的数据:\n", X_encoded)

# 使用 LabelEncoder
encoder2 = LabelEncoder()
y_encoded = encoder2.fit_transform(X)


print("编码后的标签:", y_encoded)