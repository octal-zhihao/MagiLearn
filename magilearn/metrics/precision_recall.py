import numpy as np


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
    # Convert y_true and y_pred to NumPy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate true positives and false positives
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    # Avoid division by zero
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
    # Convert y_true and y_pred to NumPy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate true positives and false negatives
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    # Avoid division by zero
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
