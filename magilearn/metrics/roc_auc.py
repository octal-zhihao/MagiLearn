import numpy as np

def roc_auc_score(y_true, y_scores):
    """
    计算ROC AUC分数。

    Parameters:
    y_true (array-like): 真实的二元标签。
    y_scores (array-like): 预测分数或概率。

    Return:
    float: ROC AUC分数，表示ROC曲线下面积。
    """
    # 确保 y_true 和 y_scores 是 NumPy 数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 找到降序排列的唯一阈值
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tprs = []
    fprs = []

    # 遍历每个阈值
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # 将列表转换为 NumPy 数组
    tprs, fprs = np.array(tprs), np.array(fprs)

    # 添加 (0,0) 和 (1,1) 到 tprs 和 fprs 以形成完整的曲线
    tprs = np.concatenate(([0], tprs, [1]))
    fprs = np.concatenate(([0], fprs, [1]))

    # 按 fprs 排序 fprs 和 tprs 以确保正确计算 AUC
    sorted_indices = np.argsort(fprs)
    fprs = fprs[sorted_indices]
    tprs = tprs[sorted_indices]

    # 使用梯形法则计算 AUC
    auc = np.trapz(tprs, fprs)
    return auc
