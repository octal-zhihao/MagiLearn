import numpy as np
import pandas as pd

def load_iris():
    """
    加载 Iris 数据集。

    返回：
        dict: 包含特征、标签、特征名称和标签名称的字典。
    """
    # 数据集路径（这里假设数据文件为 iris.csv）
    filepath = 'iris.csv'  # 请确保文件路径正确
    # 读取 CSV 文件
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"无法找到文件: {filepath}")
    # 特征数据
    X = data.iloc[:, :-1].values  # 获取前四列作为特征
    # 标签数据
    y = data.iloc[:, -1].values    # 获取最后一列作为标签
    # 转换标签为数字
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    try:
        y = np.array([species_map[species] for species in y])
    except KeyError as e:
        raise ValueError(f"未识别的标签: {e}")
    # 特征名称
    feature_names = data.columns[:-1].tolist()
    # 标签名称
    target_names = list(species_map.keys())
    # 返回数据
    return {
        'data': X,
        'target': y,
        'feature_names': feature_names,
        'target_names': target_names,
    }
