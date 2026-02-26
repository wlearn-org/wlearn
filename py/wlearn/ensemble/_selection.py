"""Caruana greedy ensemble selection matching JS @wlearn/ensemble/selection.js."""

import numpy as np

from ..errors import ValidationError
from ..automl._cv import get_scorer


def caruana_select(oof_predictions, y_true, max_size=20, scoring='accuracy',
                   task='classification', n_classes=0, refine_weights=True):
    """Greedy ensemble selection (Caruana et al., 2004).

    Selects a weighted subset from a pool of OOF predictions by greedily
    adding the candidate that most improves the ensemble score at each step.
    Candidates can be selected multiple times (with replacement).

    Args:
        oof_predictions: list of np.ndarray per candidate
            Classification: each (n * n_classes,) flat row-major proba
            Regression: each (n,)
        y_true: np.ndarray
        max_size: max ensemble members
        scoring: metric name or callable
        task: 'classification' or 'regression'
        n_classes: inferred from data if 0
        refine_weights: if True, optimize weights via projected gradient descent
            after Caruana selection (requires optimize_weights from _weights.py)

    Returns:
        dict with 'indices', 'weights', 'scores'
    """
    n = len(y_true)
    n_candidates = len(oof_predictions)

    if n_candidates == 0:
        raise ValidationError('caruana_select: need at least 1 candidate')

    scorer_fn = get_scorer(scoring)

    pred_size = len(oof_predictions[0]) / n
    if pred_size != int(pred_size):
        raise ValidationError(
            'caruana_select: oof_predictions[0].length must be divisible by n'
        )
    pred_size = int(pred_size)

    if task == 'classification' and n_classes == 0:
        n_classes = pred_size

    # Current ensemble prediction (running weighted average)
    current = np.zeros(len(oof_predictions[0]), dtype=np.float64)
    selected = []
    scores = []

    for t in range(max_size):
        best_idx = -1
        best_score = -float('inf')

        for i in range(n_candidates):
            trial = _trial_predictions(current, oof_predictions[i], t, t + 1)
            trial_score = _score(trial, y_true, scorer_fn, task, n_classes, n)
            if trial_score > best_score:
                best_score = trial_score
                best_idx = i

        selected.append(best_idx)
        scores.append(best_score)

        # Update running ensemble
        P = oof_predictions[best_idx]
        for j in range(len(current)):
            current[j] = (t * current[j] + P[j]) / (t + 1)

    # Compute weights from selection counts
    counts = {}
    for idx in selected:
        counts[idx] = counts.get(idx, 0) + 1

    unique_indices = np.array(sorted(counts.keys()), dtype=np.int32)
    weights = np.array(
        [counts[int(idx)] / max_size for idx in unique_indices],
        dtype=np.float64
    )

    result = {
        'indices': unique_indices,
        'weights': weights,
        'scores': np.array(scores, dtype=np.float64),
    }

    if refine_weights and len(unique_indices) > 1:
        from ._weights import optimize_weights
        selected_oofs = [oof_predictions[int(idx)] for idx in unique_indices]
        refined = optimize_weights(
            selected_oofs, y_true, weights, task=task,
        )
        result['weights'] = refined

    return result


def _trial_predictions(current, candidate, t_count, t_total):
    trial = np.empty(len(current), dtype=np.float64)
    for j in range(len(current)):
        trial[j] = (t_count * current[j] + candidate[j]) / t_total
    return trial


def _score(preds, y_true, scorer_fn, task, n_classes, n):
    if task == 'regression':
        return scorer_fn(y_true, preds)
    # Classification: convert proba to hard predictions via argmax
    hard_preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        best_c = 0
        best_v = -float('inf')
        for c in range(n_classes):
            if preds[i * n_classes + c] > best_v:
                best_v = preds[i * n_classes + c]
                best_c = c
        hard_preds[i] = best_c
    return scorer_fn(y_true, hard_preds)
