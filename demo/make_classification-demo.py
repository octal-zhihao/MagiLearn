# 导入 make_classification 函数
from magilearn.datasets.make_classification import make_classification
# from sklearn.datasets import make_classification

# 生成数据集
# X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=2,
#                            n_clusters_per_class=2, random_state=42)

X, y = make_classification(
    n_samples=1000,         # 样本数
    n_features=20,          # 特征数
    n_classes=2,            # 类别数
    random_state=42         # 随机种子
)

# 打印特征和标签的形状
print("特征矩阵形状:", X.shape)
print("标签数组形状:", y.shape)

# 查看前5个样本
print("特征矩阵（前5行）:\n", X[:5])
print("标签（前5个）:", y[:5])
