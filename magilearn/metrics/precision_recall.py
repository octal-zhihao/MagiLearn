import numpy as np


def precision_score(y_true, y_pred):
    """
    计算精确率。

    Parameters:
    y_true (array-like): 真实标签。
    y_pred (array-like): 预测标签。

    Return
    float: 精确率，表示所有正预测中真实为正的比例。
    """
    # 将 y_true 和 y_pred 转换为 NumPy 数组，以便进行逐元素操作
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算真正例和假正例
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    # 避免除以零
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


def recall_score(y_true, y_pred):
    """
    计算召回率。

    Parameters:
    y_true (array-like): 真实标签。
    y_pred (array-like): 预测标签。

    Return
    float: 召回率，表示所有实际为正的实例中真正为正的比例。
    """
    # 将 y_true 和 y_pred 转换为 NumPy 数组，以便进行逐元素操作
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算真正例和假负例
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    # 避免除以零
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
