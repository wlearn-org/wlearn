"""High-level AutoML matching JS automl/auto-fit.js."""

import numpy as np

from ..errors import ValidationError
from ..ensemble._voting import VotingEnsemble
from ..ensemble._oof import get_oof_predictions
from ..ensemble._selection import caruana_select
from ._search import RandomSearch, SuccessiveHalvingSearch
from ._portfolio import PortfolioSearch
from ._common import detect_task


def _normalize_specs(models):
    """Normalize model specs: accept dicts or (name, cls, params) tuples."""
    out = []
    for m in models:
        if isinstance(m, (list, tuple)):
            out.append({
                'name': m[0],
                'cls': m[1],
                'params': m[2] if len(m) > 2 else {},
            })
        else:
            out.append(m)
    return out


def auto_fit(models, X, y, ensemble=False, ensemble_size=20, refit=True,
             scoring=None, cv=5, seed=42, task=None, n_iter=20, max_time_ms=0,
             strategy='random'):
    """High-level AutoML: search + optional Caruana ensemble + refit.

    Args:
        models: list of model specs (dicts or tuples)
        X: np.ndarray feature matrix
        y: np.ndarray labels
        ensemble: whether to build a Caruana ensemble
        ensemble_size: max ensemble members
        refit: whether to refit best model on full data
        scoring: metric name or None (auto-detect)
        cv: number of CV folds
        seed: random seed
        task: 'classification' or 'regression' or None (auto-detect)
        n_iter: candidates per model (for random/halving strategies)
        max_time_ms: time limit
        strategy: 'random' (default), 'portfolio', or 'halving'

    Returns:
        dict with: model, leaderboard, bestParams, bestModelName, bestScore
    """
    specs = _normalize_specs(models)
    if not specs:
        raise ValidationError('auto_fit: at least one model is required')

    if strategy == 'portfolio':
        search = PortfolioSearch(
            specs, scoring=scoring, cv=cv, seed=seed, task=task,
            max_time_ms=max_time_ms,
        )
    elif strategy == 'halving':
        search = SuccessiveHalvingSearch(
            specs, scoring=scoring, cv=cv, seed=seed, task=task,
            n_iter=n_iter, max_time_ms=max_time_ms,
        )
    else:
        search = RandomSearch(
            specs, scoring=scoring, cv=cv, seed=seed, task=task,
            n_iter=n_iter, max_time_ms=max_time_ms,
        )
    result = search.fit(X, y)
    leaderboard = result['leaderboard']
    best_result = result['bestResult']
    ranked = leaderboard.ranked()

    task_actual = task or detect_task(y)
    scoring_actual = scoring or ('accuracy' if task_actual == 'classification' else 'r2')

    model = None

    if ensemble:
        # Select best config per model family
        family_best = {}
        for entry in ranked:
            if entry['modelName'] not in family_best:
                family_best[entry['modelName']] = entry

        # Build pool: best per family + top remaining up to ensemble_size * 2
        pool = list(family_best.values())
        pool_ids = set(e['id'] for e in pool)
        for entry in ranked:
            if len(pool) >= ensemble_size * 2:
                break
            if entry['id'] not in pool_ids:
                pool.append(entry)
                pool_ids.add(entry['id'])

        # Map model names to classes
        cls_map = {spec['name']: spec['cls'] for spec in specs}

        # Build estimator specs for OOF
        est_specs = [
            (f"{entry['modelName']}_{i}", cls_map[entry['modelName']], entry['params'])
            for i, entry in enumerate(pool)
        ]

        # Generate OOF predictions
        oof_result = get_oof_predictions(est_specs, X, y, cv=cv, seed=seed, task=task_actual)
        oof_preds = oof_result['oofPreds']

        # Caruana selection
        sel = caruana_select(
            oof_preds, y,
            max_size=ensemble_size,
            scoring=scoring_actual,
            task=task_actual,
        )

        # Build VotingEnsemble from selected
        selected_specs = [est_specs[int(i)] for i in sel['indices']]
        selected_weights = sel['weights']

        ens = VotingEnsemble.create(
            estimators=selected_specs,
            weights=list(selected_weights),
            voting='soft' if task_actual == 'classification' else 'soft',
            task=task_actual,
        )
        ens.fit(X, y)
        model = ens

    elif refit:
        model = search.refit_best(X, y)

    return {
        'model': model,
        'leaderboard': ranked,
        'bestParams': best_result['params'],
        'bestModelName': best_result['modelName'],
        'bestScore': best_result['meanScore'],
    }
