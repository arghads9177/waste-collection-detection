"""Classification metrics: accuracy, macro-F1, per-class P/R/F1, confusion matrix."""
import logging
from typing import Any, Sequence

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Class names in index order (defines label set and matrix axes).

    Returns:
        Dict with keys: accuracy, macro_f1, per_class (name -> {precision, recall,
        f1, support}), confusion_matrix (list of lists, rows=true, cols=pred).
    """
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()) if len(f1) > 0 else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
