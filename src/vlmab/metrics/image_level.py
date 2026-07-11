"""Image-level metrics: I-AUROC, I-AP, I-F1max. Pure numpy/sklearn, unit-tested."""
import numpy as np
from sklearn import metrics as skm


def i_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(skm.roc_auc_score(labels, scores))


def i_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(skm.average_precision_score(labels, scores))


def i_f1max(labels: np.ndarray, scores: np.ndarray) -> float:
    prec, rec, _ = skm.precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(np.nanmax(f1))
