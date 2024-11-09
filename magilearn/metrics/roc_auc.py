# roc_auc.py
# This script defines a function for calculating the ROC AUC (Receiver Operating Characteristic - Area Under Curve),
# a metric used to evaluate the performance of binary classification models based on their true positive rate (TPR)
# and false positive rate (FPR) across different decision thresholds.
# The AUC represents the area under the ROC curve, providing a single value summary
# of the model's ability to distinguish between positive and negative classes.

import numpy as np


def calculate_roc_auc(y_true, y_scores):
    """
    Calculate the ROC AUC score.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Predicted scores or probabilities.

    Returns:
    float: ROC AUC score, representing the area under the ROC curve.
    """
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
    auc = np.trapz(tprs, fprs)  # Calculate AUC
    return auc
