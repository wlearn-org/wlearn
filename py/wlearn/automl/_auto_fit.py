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
from ._bayesian import BayesianSearch
from ._common import detect_task
from ._candidate import class_for_candidate, normalize_model_specs
from ._candidate_pipeline import create_candidate_pipeline_class
from ..preprocess import (
    TYPE_ID as PREPROCESS_TYPE_ID,
    resolve_preprocess_config,
)


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


def _resolve_preprocess_choices(value):
    if value is False or value is None:
        return [None]
    if value is True:
        return [_resolved_template('wlearn.preprocess.default.v1', {})]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValidationError(
                'auto_fit preprocess template list must be nonempty.')
        seen = set()
        result = []
        for index, template in enumerate(value):
            if not isinstance(template, dict):
                raise ValidationError(
                    f'preprocess[{index}] must be a template dict.')
            template_id = template.get('templateId')
            if not isinstance(template_id, str) or not template_id:
                raise ValidationError(
                    f'preprocess[{index}].templateId must be a nonempty string.')
            if template_id in seen:
                raise ValidationError(
                    f'duplicate preprocessing templateId "{template_id}".')
            seen.add(template_id)
            if template.get('typeId') != PREPROCESS_TYPE_ID:
                raise ValidationError(
                    f'preprocess[{index}].typeId must be '
                    f'"{PREPROCESS_TYPE_ID}".')
            if 'searchSpace' in template:
                search_space = template['searchSpace']
                if not isinstance(search_space, dict):
                    raise ValidationError(
                        f'preprocess[{index}].searchSpace must be a dict.')
                if search_space:
                    raise ValidationError(
                        'preprocessing searchSpace is not supported in the '
                        'fixed-template V1; enumerate explicit templates.')
            params = template['params'] if 'params' in template else {}
            if not isinstance(params, dict):
                raise ValidationError(
                    f'preprocess[{index}].params must be a dict.')
            result.append(_resolved_template(template_id, params))
        return result
    if not isinstance(value, dict):
        raise ValidationError(
            'auto_fit preprocess must be false, true, a config dict, '
            'or a template list.')
    return [_resolved_template('wlearn.preprocess.inline.v1', value)]


def _resolved_template(template_id, config):
    return {
        'templateId': template_id,
        'typeId': PREPROCESS_TYPE_ID,
        'resolvedParams': resolve_preprocess_config(config),
    }


def auto_fit(models, X, y, ensemble=True, ensemble_size=20, refit=True,
             scoring=None, cv=5, seed=42, task=None, n_iter=20, max_time_ms=0,
             strategy='random', min_disagreement=0.05, stacking='auto',
             meta_estimator=None, preprocess=False,
             stacking_passthrough=None):
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
        n_iter: candidates per model for random/halving/progressive/bayesian strategies
        max_time_ms: time limit
        strategy: 'random' (default), 'portfolio', 'halving',
            'progressive', or 'bayesian'

    Returns:
        dict with: model, leaderboard, bestParams, bestModelName, bestScore
    """
    preprocess_choices = _resolve_preprocess_choices(preprocess)
    specs = normalize_model_specs(models, 'auto_fit models')
    specs = [{
        **spec,
        'preprocessChoices': preprocess_choices,
        'createCandidateClass': (
            lambda candidate, current=spec:
            create_candidate_pipeline_class(
                current, candidate, base_seed=seed, fold_count=cv)),
    } for spec in specs]

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
    elif strategy == 'bayesian':
        search = BayesianSearch(
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
    archive = result['archive']
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
            class_id = entry['candidate']['model']['classId']
            if class_id not in family_best:
                family_best[class_id] = entry
            elif class_id not in family_second:
                family_second[class_id] = entry

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

        spec_map = {spec['classId']: spec for spec in specs}

        # Build estimator specs for OOF
        est_specs = []
        for i, entry in enumerate(pool):
            spec = spec_map[entry['candidate']['model']['classId']]
            cls = class_for_candidate(spec, entry['candidate'])
            est_specs.append((
                f"{entry['modelName']}_{i}", cls, entry['params']))

        # Generate OOF predictions
        oof_result = get_oof_predictions(est_specs, X, y, cv=cv, seed=seed, task=task_actual)
        oof_preds = oof_result['oofPreds']

        # Disagreement filter: remove near-duplicate predictions
        filtered_idx = _filter_by_disagreement(
            oof_preds, y, task_actual, min_disagreement)
        filtered_oofs = [oof_preds[i] for i in filtered_idx]
        filtered_specs = [est_specs[i] for i in filtered_idx]
        filtered_entries = [pool[i] for i in filtered_idx]

        # Caruana selection on filtered pool
        sel = caruana_select(
            filtered_oofs, y,
            max_size=ensemble_size,
            scoring=scoring_actual,
            task=task_actual,
        )

        # Build ensemble from selected
        selected_specs = [filtered_specs[int(i)] for i in sel['indices']]
        selected_entries = [
            filtered_entries[int(i)] for i in sel['indices']]
        selected_weights = sel['weights']

        # Determine if two-layer stacking should be used
        selected_families = set(
            entry['candidate']['model']['classId']
            for entry in selected_entries)
        use_stacking = (stacking is True or
                        (stacking == 'auto' and len(selected_families) >= 3
                         and meta_estimator is not None))

        if use_stacking and meta_estimator is not None:
            if isinstance(meta_estimator, (list, tuple)):
                meta_spec = meta_estimator
            else:
                meta_spec = ('meta', meta_estimator.get('cls', meta_estimator),
                             meta_estimator.get('params', {}))
            selected_preprocessing = any(
                entry['candidate']['preprocess'] is not None
                for entry in selected_entries)
            if selected_preprocessing and stacking_passthrough is True:
                raise ValidationError(
                    'stacking_passthrough=True is invalid when '
                    'preprocessing is active.')
            ens = StackingEnsemble.create(
                estimators=selected_specs,
                final_estimator=meta_spec,
                passthrough=(
                    False if selected_preprocessing else
                    True if stacking_passthrough is None else
                    stacking_passthrough),
                task=task_actual,
                cv=cv,
                seed=seed,
            )
            model = _fit_owned_ensemble(ens, X, y)
        else:
            ens = VotingEnsemble.create(
                estimators=selected_specs,
                weights=list(selected_weights),
                voting='soft',
                task=task_actual,
            )
            model = _fit_owned_ensemble(ens, X, y)

    elif refit:
        model = search.refit_best(X, y)

    return {
        'model': model,
        'preprocessor': None,
        'leaderboard': ranked,
        'archive': archive,
        'bestParams': {
            'model': best_result['candidate']['model']['params'],
            'preprocess': (
                None if best_result['candidate']['preprocess'] is None else
                best_result['candidate']['preprocess']['resolvedParams']),
        },
        'bestCandidate': best_result['candidate'],
        'bestModelName': best_result['modelName'],
        'bestScore': best_result['meanScore'],
    }


def _fit_owned_ensemble(ensemble, X, y):
    try:
        ensemble.fit(X, y)
        return ensemble
    except Exception:
        try:
            ensemble.dispose()
        except Exception:
            pass
        raise
