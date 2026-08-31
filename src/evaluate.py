"""Dependent-variable calculations: PR-AUC (primary), fraud-class F1, ROC-AUC.

All metrics treat the illicit/fraud class as the positive class. The F1
threshold is selected on the validation set only, then applied unchanged
to the test set (proposal: Experimental Assessment)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def select_threshold(y_true_val, y_score_val):
    """Pick the classification threshold that maximizes fraud-class F1 on
    the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true_val, y_score_val)
    # precision/recall have one more entry than thresholds; drop the last point
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, y_score, threshold):
    """Return the three dependent variables for one run on the test set."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
    }
