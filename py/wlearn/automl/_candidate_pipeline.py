"""Fold-owned Pipeline construction for resolved preprocessing candidates."""

from ..pipeline import Pipeline
from ..preprocess import Preprocessor
from ..task import validate_estimator_task
from ..errors import ValidationError
from ._candidate import class_for_candidate, make_candidate_id, seed_for


def create_candidate_pipeline_class(
        spec, candidate, *, preprocessor_cls=Preprocessor,
        pipeline_cls=Pipeline, base_seed=None, fold_count=None):
    """Return a class whose create() owns preprocessor, model, then Pipeline."""
    model_cls = spec['cls']
    resolved_config = candidate['preprocess']['resolvedParams']
    provenance = _candidate_provenance(
        candidate, base_seed=base_seed, fold_count=fold_count)

    class PreprocessedModel:
        class_id = spec['classId']

        @classmethod
        def create(cls, params=None):
            preprocessor = preprocessor_cls(resolved_config)
            estimator = None
            try:
                estimator = model_cls.create(params or {})
                return pipeline_cls([
                    ('preprocess', preprocessor),
                    ('model', estimator),
                ], provenance=provenance)
            except Exception:
                _dispose_quietly(estimator)
                _dispose_quietly(preprocessor)
                raise

        @classmethod
        def default_search_space(cls, task=None):
            from ._common import default_search_space
            return default_search_space({'cls': model_cls, 'params': {'task': task}})

        @classmethod
        def budget_spec(cls):
            method = getattr(model_cls, 'budget_spec', None)
            return method() if method is not None else None

    PreprocessedModel.__name__ = (
        f'{getattr(model_cls, "__name__", spec["name"])}WithPreprocessing')
    return PreprocessedModel


def _candidate_provenance(candidate, *, base_seed, fold_count):
    provenance = {
        'candidateId': make_candidate_id(candidate),
        'candidate': candidate,
    }
    if base_seed is None and fold_count is None:
        return provenance
    effective_seed = 42 if base_seed is None else base_seed
    effective_folds = 5 if fold_count is None else fold_count
    if (isinstance(effective_folds, bool) or
            not isinstance(effective_folds, int) or effective_folds < 1):
        raise ValidationError('fold_count must be a positive integer.')
    provenance['baseSeed'] = effective_seed
    provenance['foldSeeds'] = [
        {'foldId': fold_id,
         'seed': seed_for(candidate, fold_id, effective_seed)}
        for fold_id in range(effective_folds)
    ]
    return provenance


def _dispose_quietly(value):
    if value is None or not hasattr(value, 'dispose'):
        return
    try:
        value.dispose()
    except Exception:
        pass


def fit_candidate(spec, candidate, X, y, candidate_id=None):
    """Create and fit one candidate, disposing it if fitting fails."""
    if (candidate_id is not None and
            make_candidate_id(candidate) != candidate_id):
        raise ValidationError(
            'candidate_id does not match the structured candidate.')
    candidate_cls = class_for_candidate(spec, candidate)
    instance = candidate_cls.create(candidate['model']['params'])
    try:
        instance.fit(X, y)
        validate_estimator_task(instance, candidate['model']['params'].get('task'))
        return instance
    except Exception:
        _dispose_quietly(instance)
        raise
