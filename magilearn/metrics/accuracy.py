# accuracy.py
# This script defines a function for calculating the accuracy score,
# a metric commonly used to evaluate the performance of classification models.
# The function compares true labels with predicted labels and returns
# the proportion of correct predictions.

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.

    Returns:
    float: Accuracy score, representing the proportion of correct predictions.
    """
    correct = sum(y_true == y_pred)
    return correct / len(y_true)
