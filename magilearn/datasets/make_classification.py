import numpy as np
def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                        n_clusters_per_class=1, n_classes=2, weights=None, random_state=None, class_sep=1.0):
    """
    生成一个用于分类的随机数据集，增强类别之间的分离性。
    参数：
        n_samples (int): 样本数量。
        n_features (int): 特征总数。
        n_informative (int): 信息性特征数量。
        n_redundant (int): 冗余特征数量。
        n_clusters_per_class (int): 每个类别的聚类数量。
        n_classes (int): 数据集中的类别数量。
        weights (list of float, optional): 每个类别的样本比例。
        random_state (int, optional): 随机种子。
        class_sep (float): 类别间隔的大小，用于控制类别中心的分离程度。
    返回：
        X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)。
        y (numpy.ndarray): 标签数组，形状为 (n_samples,)。
    """
    if random_state is not None:
        np.random.seed(random_state)
    # 处理权重
    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    elif sum(weights) != 1.0:
        weights = np.array(weights) / sum(weights)  # 归一化
    # 确定每个类别的样本数量
    n_samples_per_class = (np.array(weights) * n_samples).astype(int)
    n_samples_per_class[-1] = n_samples - n_samples_per_class[:-1].sum()  # 确保总样本数一致
    X_list = []
    y_list = []
    # 生成每个类别的数据
    for class_id, samples in enumerate(n_samples_per_class):
        # 生成信息性特征并添加类别间隔
        informative_features = np.random.randn(samples, n_informative) + (class_id * class_sep)
        # 生成冗余特征（由信息性特征的线性组合得到）
        redundant_features = np.dot(informative_features, np.random.randn(n_informative, n_redundant))
        # 生成随机特征
        n_random = n_features - n_informative - n_redundant
        random_features = np.random.randn(samples, n_random)
        # 合并特征
        X_class = np.hstack((informative_features, redundant_features, random_features))
        # 为每个类别生成多个簇
        if n_clusters_per_class > 1:
            for cluster_id in range(1, n_clusters_per_class):
                cluster_shift = np.random.randn(1, n_features) * 0.1  # 簇的偏移
                mask = np.random.rand(samples) < (1 / n_clusters_per_class)  # 随机选取部分样本
                X_class[mask] += cluster_shift
        # 添加类别数据
        X_list.append(X_class)
        y_list.append(np.full(samples, class_id))
    # 合并所有类别的数据
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # 打乱数据顺序
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


