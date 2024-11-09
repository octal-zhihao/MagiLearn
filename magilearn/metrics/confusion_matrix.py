# confusion_matrix.py
# This script defines a function for calculating the confusion matrix,
# a table often used to describe the performance of a classification model.
# The confusion matrix compares true labels with predicted labels and
# provides counts of true positives, true negatives, false positives, and false negatives
# for each class in a multi-class classification problem.

import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    numpy.ndarray: Confusion matrix where rows represent actual classes
                   and columns represent predicted classes.
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i, j] = sum((y_true == actual) & (y_pred == predicted))
    return matrix
