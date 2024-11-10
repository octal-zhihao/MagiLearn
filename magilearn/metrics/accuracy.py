import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.

    Returns:
    float: Accuracy score, representing the proportion of correct predictions.
    """
    # Convert y_true and y_pred to NumPy arrays to ensure element-wise comparison
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length")

    # Perform element-wise comparison and count correct predictions
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)
