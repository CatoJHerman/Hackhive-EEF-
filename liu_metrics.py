"""Binary sample-level evaluation with explicit undefined metrics."""
from sklearn.metrics import confusion_matrix


def metrics(y, predicted):
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "fault_recall": tp / (tp + fn) if tp + fn else None,
            "missed_fault_rate": fn / (tp + fn) if tp + fn else None}
