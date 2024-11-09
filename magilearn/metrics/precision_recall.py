# precision_recall.py
# This script defines functions for calculating precision and recall scores,
# two metrics commonly used to evaluate the performance of binary classification models.
# Precision measures the accuracy of positive predictions, while recall measures the
# ability of the model to find all positive instances.

def precision_score(y_true, y_pred):
    """
    Calculate the precision score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Precision score, representing the proportion of true positives
           out of all positive predictions.
    """
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_positive = sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def recall_score(y_true, y_pred):
    """
    Calculate the recall score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Recall score, representing the proportion of true positives
           out of all actual positive instances.
    """
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_negative = sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
