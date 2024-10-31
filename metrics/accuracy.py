# accuracy.py

def calculate_accuracy(y_true, y_pred):
    correct = sum(y_true == y_pred)
    return correct / len(y_true)
