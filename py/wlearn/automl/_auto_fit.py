"""High-level AutoML matching JS automl/auto-fit.js."""

import math
import numpy as np

from ..errors import ValidationError
from ..ensemble._voting import VotingEnsemble
from ..ensemble._stacking import StackingEnsemble
from ..ensemble._oof import get_oof_predictions
from ..ensemble._selection import caruana_select
from ._search import RandomSearch, SuccessiveHalvingSearch
from ._portfolio import PortfolioSearch
from ._progressive import ProgressiveSearch
from ._common import detect_task
from ..preprocess import Preprocessor


def _disagreement_rate(a, b, n, task):
    """Pairwise disagreement rate between two prediction vectors."""
    if task == 'classification':
        n_classes = len(a) // n
        disagree = 0
        for i in range(n):
            best_a = 0
            best_b = 0
            best_va = -float('inf')
            best_vb = -float('inf')
            for c in range(n_classes):
                idx = i * n_classes + c
                if a[idx] > best_va:
                    best_va = a[idx]
                    best_a = c
                if b[idx] > best_vb:
                    best_vb = b[idx]
                    best_b = c
            if best_a != best_b:
                disagree += 1
        return disagree / n
    # Regression: 1 - abs(correlation)
    sum_a = sum_b = sum_aa = sum_bb = sum_ab = 0.0
    for i in range(n):
        sum_a += a[i]
        sum_b += b[i]
        sum_aa += a[i] * a[i]
        sum_bb += b[i] * b[i]
        sum_ab += a[i] * b[i]
    denom = math.sqrt((sum_aa - sum_a * sum_a / n) *
                      (sum_bb - sum_b * sum_b / n))
    if denom < 1e-12:
        return 1.0
    corr = (sum_ab - sum_a * sum_b / n) / denom
    return 1.0 - abs(corr)


def _filter_by_disagreement(oof_preds, y, task, min_disagreement):
    """Filter pool by minimum pairwise disagreement."""
    n = len(y)
    if len(oof_preds) <= 2 or min_disagreement <= 0:
        return list(range(len(oof_preds)))
    kept = [0]
    for i in range(1, len(oof_preds)):
        diverse = True
        for j in kept:
            if _disagreement_rate(oof_preds[i], oof_preds[j], n, task) < min_disagreement:
                diverse = False
                break
        if diverse:
            kept.append(i)
    if len(kept) < 2 and len(oof_preds) >= 2:
        if 1 not in kept:
            kept.append(1)
    return kept


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


def auto_fit(models, X, y, ensemble=True, ensemble_size=20, refit=True,
             scoring=None, cv=5, seed=42, task=None, n_iter=20, max_time_ms=0,
             strategy='random', min_disagreement=0.05, stacking='auto',
             meta_estimator=None, preprocess=False):
    """High-level AutoML: search + optional Caruana ensemble + refit.

    Args:
        models: list of model specs (dicts or tuples)
        X: np.ndarray feature matrix
        y: np.ndarray labels
        ensemble: whether to build a Caruana ensemble (default True)
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

    # Optional preprocessing
    preprocessor = None
    if preprocess:
        pp_config = preprocess if isinstance(preprocess, dict) else {}
        preprocessor = Preprocessor(**pp_config)
        X = preprocessor.fit_transform(X, y)

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
    elif strategy == 'progressive':
        search = ProgressiveSearch(
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
        # Diversity-aware pool: best per family + top overall
        family_best = {}
        family_second = {}
        for entry in ranked:
            if entry['modelName'] not in family_best:
                family_best[entry['modelName']] = entry
            elif entry['modelName'] not in family_second:
                family_second[entry['modelName']] = entry

        # Seed pool: best per family (guaranteed diversity)
        pool = list(family_best.values())
        pool_ids = set(e['id'] for e in pool)

        # Add second-best per family
        for entry in family_second.values():
            if len(pool) >= ensemble_size * 2:
                break
            if entry['id'] not in pool_ids:
                pool.append(entry)
                pool_ids.add(entry['id'])

        # Fill remaining from top overall
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

        # Disagreement filter: remove near-duplicate predictions
        filtered_idx = _filter_by_disagreement(
            oof_preds, y, task_actual, min_disagreement)
        filtered_oofs = [oof_preds[i] for i in filtered_idx]
        filtered_specs = [est_specs[i] for i in filtered_idx]

        # Caruana selection on filtered pool
        sel = caruana_select(
            filtered_oofs, y,
            max_size=ensemble_size,
            scoring=scoring_actual,
            task=task_actual,
        )

        # Build ensemble from selected
        selected_specs = [filtered_specs[int(i)] for i in sel['indices']]
        selected_weights = sel['weights']

        # Determine if two-layer stacking should be used
        selected_families = set(s[0].split('_')[0] for s in selected_specs)
        use_stacking = (stacking is True or
                        (stacking == 'auto' and len(selected_families) >= 3
                         and meta_estimator is not None))

        if use_stacking and meta_estimator is not None:
            if isinstance(meta_estimator, (list, tuple)):
                meta_spec = meta_estimator
            else:
                meta_spec = ('meta', meta_estimator.get('cls', meta_estimator),
                             meta_estimator.get('params', {}))
            ens = StackingEnsemble.create(
                estimators=selected_specs,
                final_estimator=meta_spec,
                passthrough=True,
                task=task_actual,
                cv=cv,
                seed=seed,
            )
            ens.fit(X, y)
            model = ens
        else:
            ens = VotingEnsemble.create(
                estimators=selected_specs,
                weights=list(selected_weights),
                voting='soft',
                task=task_actual,
            )
            ens.fit(X, y)
            model = ens

    elif refit:
        model = search.refit_best(X, y)

    return {
        'model': model,
        'preprocessor': preprocessor,
        'leaderboard': ranked,
        'bestParams': best_result['params'],
        'bestModelName': best_result['modelName'],
        'bestScore': best_result['meanScore'],
    }
