"""
Metrics shared by the pixel- and pair-level evaluation (same definitions as the
original evaluation scripts used for the paper).
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float,
                           inclusive: bool = False):
    preds = scores >= threshold if inclusive else scores > threshold
    tp = int((preds & labels).sum())
    fp = int((preds & ~labels).sum())
    fn = int((~preds & labels).sum())
    tn = int((~preds & ~labels).sum())
    return tp, fp, fn, tn


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold=None,
                   inclusive: bool = False) -> dict:
    """
    AUC / AP (threshold-free) and precision / recall / F1 / FPR of the positive
    class at `threshold` (positive: score > threshold, or >= when `inclusive`).
    Threshold metrics are NaN when no threshold is given; AUC / AP are NaN when
    only one class is present. Infinite scores (TSED) are replaced by
    10 × the largest finite score + 1 for AUC / AP.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    m = dict(n=len(labels), n_pos=n_pos, n_neg=n_neg,
             auc=float("nan"), ap=float("nan"),
             threshold=float("nan") if threshold is None else float(threshold),
             precision=float("nan"), recall=float("nan"), f1=float("nan"), fpr=float("nan"),
             tp=None, fp=None, fn=None, tn=None)
    if n_pos > 0 and n_neg > 0:
        scores_fi = scores
        is_inf = np.isinf(scores)
        if is_inf.any():
            finite = scores[~is_inf]
            inf_sub = (finite.max() if len(finite) else 1.0) * 10 + 1.0
            scores_fi = np.where(is_inf, inf_sub, scores)
        m["auc"] = float(roc_auc_score(labels, scores_fi))
        m["ap"] = float(average_precision_score(labels, scores_fi))
    if threshold is not None:
        tp, fp, fn, tn = confusion_at_threshold(scores, labels, threshold, inclusive)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        m.update(precision=float(p), recall=float(r),
                 f1=float(2 * p * r / max(p + r, 1e-8)),
                 fpr=float(fp / max(fp + tn, 1)),
                 tp=tp, fp=fp, fn=fn, tn=tn)
    return m
