# confusion_matrix.py
# 本脚本定义了一个计算混淆矩阵的函数，
# 混淆矩阵是描述分类模型性能的表格。
# 混淆矩阵将真实标签与预测标签进行比较，
# 并提供每个类的真正例、真负例、假正例和假负例的计数
# 适用于多类别分类问题。

import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵。

    Parameters:
    y_true (array-like): 真实标签。
    y_pred (array-like): 预测标签。

    Returns:
    numpy.ndarray: 混淆矩阵，其中行表示实际类别，
                   列表示预测类别。
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i, j] = sum((y_true == actual) & (y_pred == predicted))
    return matrix
