import numpy as np

def accuracy_score(y_true, y_pred):
    """
    计算准确率。

    Parameters:
    y_true (list 或 array-like): 真实标签。
    y_pred (list 或 array-like): 预测标签。

    Returns:
    float: 准确率，表示正确预测的比例。
    """
    # 将 y_true 和 y_pred 转换为 NumPy 数组，以确保逐元素比较
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 确保 y_true 和 y_pred 长度相同
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须具有相同的长度")

    # 执行逐元素比较并统计正确预测的数量
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)
