import numpy as np


def calculate_accracy(y_true, y_pred):
    # (tp + tn) / total
    correct_pred = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct_pred / total


def calculate_precision(y_true, y_pred):
    # tp / (tp + fp)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def calculate_recall(y_true, y_pred):
    # tp / (tp + fn)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def calculate_f1_score(y_true, y_pred):
    # 2 * ((precision * recall) / (precision + recall))
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return (
        2 * ((precision * recall) / (precision + recall))
        if (precision + recall) != 0
        else 0
    )


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

    print("Accuracy: ", calculate_accracy(y_true, y_pred))
    print("Precision: ", calculate_precision(y_true, y_pred))
    print("Recall: ", calculate_recall(y_true, y_pred))
    print("F1-score: ", calculate_f1_score(y_true, y_pred))
