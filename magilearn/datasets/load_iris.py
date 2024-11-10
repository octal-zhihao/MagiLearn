import os
import numpy as np

def load_iris(return_X_y=False):
    """
    加载并返回鸢尾花数据集（与 sklearn 一致的结构），数据从 `datasets/data/` 路径中的多个文件中读取。
    
    参数:
    - return_X_y (bool): 如果为 True，返回 (X, y) 而不是字典格式。
    
    返回值:
    - 如果 return_X_y=False，返回字典格式，包含 'data', 'target', 'target_names', 'feature_names', 'DESCR'。
    - 如果 return_X_y=True，返回 (X, y) 元组。
    """

    # 获取当前文件所在目录，并指定数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    # 定义文件列表（包含所有要读取的文件名）
    file_names = ['bezdekIris.data', 'iris.data', 'iris.data']  # 确认文件名无误

    # 特征名称和目标名称
    feature_names = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
    target_names = np.array(['setosa', 'versicolor', 'virginica'])

    # 初始化数据容器
    data = []
    target = []

    # 读取每个文件
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 忽略空行
                    parts = line.split(',')
                    # 最后一个是类别
                    features = list(map(float, parts[:-1]))  # 转换为浮点数
                    label = parts[-1]
                    
                    # 将类别标签转换为整数
                    if label == 'Iris-setosa':
                        target.append(0)
                    elif label == 'Iris-versicolor':
                        target.append(1)
                    elif label == 'Iris-virginica':
                        target.append(2)
                    
                    data.append(features)

    # 转换为 numpy 数组
    data = np.array(data)
    target = np.array(target)

    if return_X_y:
        return data, target

    # 返回数据集
    return {
        'data': data,
        'target': target,
        'target_names': target_names,
        'feature_names': feature_names,
        'DESCR': 'Iris dataset: A classic dataset used for classification problems.'
    }
