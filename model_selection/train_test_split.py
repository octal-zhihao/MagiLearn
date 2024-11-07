import numpy as np

def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    将数据集 X 和 y 拆分为训练集和测试集。
    
    Parameters:
    - X: 特征矩阵
    - y: 标签向量
    - test_size: 测试集的比例（0 到 1 之间的浮动值）
    - random_state: 随机种子，保证划分的可复现性
    
    Returns:
    - X_train, X_test, y_train, y_test: 划分后的训练集和测试集
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 计算样本数
    total_size = len(X)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size

    # 随机打乱数据
    indices = np.random.permutation(total_size)

    # 获取训练集和测试集的索引
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 返回划分后的数据
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
