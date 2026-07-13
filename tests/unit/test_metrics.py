"""Unit tests for classification metrics against known confusion matrices."""
import pytest

from waste_classification.training.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions():
    class_names = ["a", "b", "c"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    for cls in class_names:
        assert metrics["per_class"][cls]["precision"] == 1.0
        assert metrics["per_class"][cls]["recall"] == 1.0
        assert metrics["per_class"][cls]["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_compute_classification_metrics_known_confusion():
    class_names = ["cat", "dog"]
    # 3 cats correctly predicted, 1 cat misclassified as dog; 2 dogs correct.
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[3, 1], [0, 2]]

    cat = metrics["per_class"]["cat"]
    assert cat["precision"] == pytest.approx(1.0)  # 3/3 predicted-cat are true cats
    assert cat["recall"] == pytest.approx(0.75)  # 3/4 true cats predicted correctly
    assert cat["support"] == 4

    dog = metrics["per_class"]["dog"]
    assert dog["precision"] == pytest.approx(2 / 3)  # 2/3 predicted-dog are true dogs
    assert dog["recall"] == pytest.approx(1.0)  # 2/2 true dogs predicted correctly
    assert dog["support"] == 2


def test_compute_classification_metrics_empty_class_zero_division():
    """A class with no predictions and no support should report zeros, not raise."""
    class_names = ["x", "y", "z"]
    y_true = [0, 0]
    y_pred = [0, 0]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert metrics["per_class"]["z"]["precision"] == 0.0
    assert metrics["per_class"]["z"]["recall"] == 0.0
    assert metrics["per_class"]["z"]["f1"] == 0.0
    assert metrics["per_class"]["z"]["support"] == 0
