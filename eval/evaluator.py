import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import torch
import random


def compute_auc(predictions: np.ndarray, labels: np.ndarray, average='macro'):
    per_class = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            per_class.append(roc_auc_score(labels[:, i], predictions[:, i]))
    if len(per_class) == 0:
        return 0.0
    return float(np.mean(per_class)) if average == 'macro' else per_class


def compute_sensitivity_specificity(preds: np.ndarray, labels: np.ndarray, thr=0.5):
    binary = (preds >= thr).astype(int)
    sens = []
    spec = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            tn, fp, fn, tp = confusion_matrix(labels[:, i], binary[:, i]).ravel()
            sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(sens), np.array(spec)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, thr=0.5):
    auc = compute_auc(preds, labels)
    ap_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            ap_scores.append(average_precision_score(labels[:, i], preds[:, i]))
    mean_ap = float(np.mean(ap_scores)) if len(ap_scores) > 0 else 0.0
    sens, spec = compute_sensitivity_specificity(preds, labels, thr)
    mean_sens = float(np.mean(sens)) if len(sens) > 0 else 0.0
    mean_spec = float(np.mean(spec)) if len(spec) > 0 else 0.0
    return {
        'auc_roc': auc,
        'mean_ap': mean_ap,
        'mean_sens': mean_sens,
        'mean_spec': mean_spec,
        'per_class_sens': sens.tolist() if isinstance(sens, np.ndarray) else [],
        'per_class_spec': spec.tolist() if isinstance(spec, np.ndarray) else []
    }


def bootstrap_confidence_interval(preds: np.ndarray, labels: np.ndarray, metric_fn, n_bootstrap=1000, seed=42):
    rng = random.Random(seed)
    scores = []
    n = labels.shape[0]
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        p = preds[idx]
        l = labels[idx]
        scores.append(metric_fn(p, l))
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean = float(np.mean(scores))
    return {'mean': mean, 'ci_lower': float(lower), 'ci_upper': float(upper)}
