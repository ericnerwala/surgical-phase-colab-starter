from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, num_classes: int):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    out["per_phase_f1"] = [float(x) for x in per_f1]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    out["confusion_matrix"] = np.asarray(cm).tolist()
    return out
