# roc_auc.py

import numpy as np

def calculate_roc_auc(y_true, y_scores):
    thresholds = np.unique(y_scores)
    tprs = []
    fprs = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    tprs, fprs = np.array(tprs), np.array(fprs)
    auc = np.trapz(tprs, fprs)  # 计算AUC
    return auc
