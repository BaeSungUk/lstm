"""Fixed four-class metrics and retrospective stay/switch evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(y, pred, last) -> dict:
    y, pred, last = (np.asarray(a) for a in [y, pred, last])
    if y.ndim != 1 or not len(y) or y.shape != pred.shape or y.shape != last.shape:
        raise ValueError('Expected nonempty, equal-length one-dimensional predictions and labels.')
    if any(not np.isin(a, [0, 1, 2, 3]).all() for a in [y, pred, last]):
        raise ValueError('Class labels must be in 0..3.')
    switch = y != last
    result = {
        'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        'class_support': np.bincount(y, minlength=4).tolist(),
    }
    for name, mask in [('stay', ~switch), ('switch', switch)]:
        result[f'{name}_n'] = int(mask.sum())
        result[f'{name}_accuracy'] = float((y[mask] == pred[mask]).mean()) if mask.any() else None
    return result
