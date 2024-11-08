# confusion_matrix.py

import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i, j] = sum((y_true == actual) & (y_pred == predicted))
    return matrix
