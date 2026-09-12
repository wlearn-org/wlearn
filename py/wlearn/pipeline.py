from copy import deepcopy

from .targets import target_rows, validate_sample_weight
from .errors import ValidationError, NotFittedError, DisposedError
from .bundle import encode_bundle, validate_bundle, write_bundle_output
from .registry import (
    register, load as registry_load, _load_with_context,
    assert_required_loaders,
)

PIPELINE_TYPE_ID = 'wlearn.pipeline@1'


class Pipeline:
    """Pipeline estimator: chains transformer steps + final estimator.

    Supports fit/predict/score with transformer + estimator chains,
    as well as save/load from WLRN bundles.
    """

    def __init__(self, steps, *, provenance=None):
        """Create a pipeline from (name, estimator) pairs.

        Args:
            steps: list of (name, estimator) tuples
        """
        if not steps:
            raise ValidationError('Pipeline requires at least one step')
        self._steps = list(steps)
        self._provenance = deepcopy(provenance)
        self._fitted = False
        self._disposed = False

    def fit(self, X, y, sample_weight=None):
        """Fit the pipeline: transform through intermediates, fit last step.

        For intermediate steps that have a transform method:
        - If fit_transform exists, use it (avoids fitting + transforming separately)
        - Otherwise, call fit() then transform()
        The last step is only fitted (not transformed).
        """
        self._ensure_alive()
        if sample_weight is not None:
            sample_weight = validate_sample_weight(sample_weight, target_rows(y))
            name, final = self._steps[-1]
            if not _capabilities(final).get('sampleWeight', False):
                raise ValidationError(f'Pipeline step "{name}" does not support sample_weight')
        # Child fits mutate in place; a failed refit cannot expose the old
        # pipeline as fitted with a mixture of old and newly learned state.
        self._fitted = False
        current = X
        for i, (_, est) in enumerate(self._steps[:-1]):
            # Unweighted transforms retain their ordinary fit semantics.
            kwargs = {'sample_weight': sample_weight} if sample_weight is not None and _capabilities(est).get('sampleWeight', False) else {}
            if hasattr(est, 'fit_transform'):
                current = est.fit_transform(current, y, **kwargs)
            else:
                est.fit(current, y, **kwargs)
                current = est.transform(current)
        # Last step: fit only
        _, last = self._steps[-1]
        kwargs = {'sample_weight': sample_weight} if sample_weight is not None else {}
        last.fit(current, y, **kwargs)
        self._fitted = True
        return self

    def _transform_through(self, X):
        """Transform X through all intermediate steps (not the last)."""
        current = X
        for _, est in self._steps[:-1]:
            current = est.transform(current)
        return current

    def predict(self, X, **kwargs):
        """Transform through intermediates, predict with last step."""
        self._ensure_fitted()
        transformed = self._transform_through(X)
        _, last = self._steps[-1]
        return last.predict(transformed, **kwargs)

    def predict_proba(self, X):
        """Transform through intermediates, predict_proba with last step."""
        self._ensure_fitted()
        _, last = self._steps[-1]
        if not hasattr(last, 'predict_proba'):
            raise ValidationError('Last step does not support predict_proba')
        transformed = self._transform_through(X)
        return last.predict_proba(transformed)

    def _predict_method(self, method, X, *args, **kwargs):
        self._ensure_fitted()
        last = self._steps[-1][1]
        if not callable(getattr(last, method, None)):
            raise ValidationError(f'Last step does not support {method}')
        return getattr(last, method)(self._transform_through(X), *args, **kwargs)

    def predict_quantiles(self, X, levels):
        return self._predict_method('predict_quantiles', X, levels)

    def predict_interval(self, X, coverage=0.9):
        return self._predict_method('predict_interval', X, coverage)

    def predict_set(self, X, coverage=0.9):
        return self._predict_method('predict_set', X, coverage)

    def predict_region(self, X, coverage=0.9):
        return self._predict_method('predict_region', X, coverage)

    def predict_distribution(self, X, **kwargs):
        return self._predict_method('predict_distribution', X, **kwargs)

    def score(self, X, y):
        """Transform through intermediates, score with last step."""
        self._ensure_fitted()
        transformed = self._transform_through(X)
        _, last = self._steps[-1]
        return last.score(transformed, y)

    @classmethod
    def load(cls, data, *, loader_options=None):
        """Load a pipeline from a WLRN bundle.

        Each step's artifact is loaded via the global registry.

        Args:
            data: bytes (WLRN bundle)

        Returns:
            Pipeline instance (fitted)
        """
        manifest, _, _ = validate_bundle(data)
        actual_type_id = manifest.get('typeId')
        if actual_type_id != PIPELINE_TYPE_ID:
            raise ValidationError(
                f'Pipeline.load expected typeId "{PIPELINE_TYPE_ID}", '
                f'got "{actual_type_id}"')
        return registry_load(data, loader_options=loader_options)

    @classmethod
    def _load_from_parts(cls, manifest, toc, blobs, context):
        if manifest.get('typeId') != PIPELINE_TYPE_ID:
            raise ValidationError(
                f'Pipeline.load expected typeId "{PIPELINE_TYPE_ID}", '
                f'got "{manifest.get("typeId")}"')
        step_infos = manifest.get('steps')
        if not isinstance(step_infos, list) or not step_infos:
            raise ValidationError(
                'Pipeline manifest must contain at least one step')
        assert_required_loaders(manifest)
        steps = []
        try:
            for step_info in step_infos:
                name = step_info['name']
                entry = next((t for t in toc if t['id'] == name), None)
                if entry is None:
                    raise ValidationError(
                        f'No artifact found for pipeline step "{name}"')
                blob = bytes(
                    blobs[entry['offset']:entry['offset'] + entry['length']])
                estimator = _load_with_context(blob, context)
                steps.append((name, estimator))
            provenance = (manifest.get('metadata') or {}).get('provenance')
            pipe = cls(steps, provenance=provenance)
            pipe._fitted = True
            return pipe
        except Exception:
            _dispose_loaded([estimator for _, estimator in steps])
            raise

    def save(self, path=None):
        """Save pipeline to a WLRN bundle.

        Returns:
            bytes
        """
        self._ensure_fitted()
        # Reject incomplete children before invoking any serializer.
        for name, estimator in self._steps:
            if not callable(getattr(estimator, 'save', None)):
                raise ValidationError(
                    f'Pipeline step "{name}" does not support save()')
        manifest = {
            'typeId': PIPELINE_TYPE_ID,
            'steps': [
                {'name': name, 'params': est.get_params()
                 if hasattr(est, 'get_params') else {}}
                for name, est in self._steps
            ],
        }
        if self._provenance is not None:
            manifest['metadata'] = {
                'provenance': deepcopy(self._provenance),
            }
        artifacts = [
            {'id': name, 'data': est.save(), 'mediaType': 'application/x-wlearn-bundle'}
            for name, est in self._steps
        ]
        return write_bundle_output(encode_bundle(manifest, artifacts), path)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        first_error = None
        for _, est in reversed(self._steps):
            if hasattr(est, 'dispose'):
                try:
                    est.dispose()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def get_params(self):
        return {
            name: est.get_params() if hasattr(est, 'get_params') else {}
            for name, est in self._steps
        }

    def set_params(self, params):
        """Update named child parameters and require an explicit refit.

        Invalidation happens before the first child mutation, so a child
        ``set_params`` failure cannot leave the pipeline claiming to be fitted.
        """
        self._ensure_alive()
        if not isinstance(params, dict):
            raise ValidationError(
                'Pipeline params must be a mapping keyed by step name')
        step_names = {name for name, _ in self._steps}
        unknown = [name for name in params if name not in step_names]
        if unknown:
            raise ValidationError(
                f'Unknown Pipeline step parameter "{unknown[0]}"')
        selected = [
            (name, estimator)
            for name, estimator in self._steps
            if name in params
        ]
        for name, estimator in selected:
            if not callable(getattr(estimator, 'set_params', None)):
                raise ValidationError(
                    f'Pipeline step "{name}" does not support set_params')
        if selected:
            self._fitted = False
        for name, estimator in selected:
            estimator.set_params(params[name])
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def capabilities(self):
        """Expose an isolated snapshot of the final estimator capabilities."""
        estimator = self._steps[-1][1]
        capabilities = getattr(estimator, 'capabilities', {})
        if callable(capabilities):
            capabilities = capabilities()
        return deepcopy(capabilities)

    @property
    def classes(self):
        self._ensure_fitted()
        estimator = self._steps[-1][1]
        classes = getattr(estimator, 'classes', None)
        return classes() if callable(classes) else classes

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('Pipeline has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('Pipeline is not fitted. Call fit() first.')


def _pipeline_loader(manifest, toc, blobs, context):
    """Registry loader for wlearn.pipeline@1 bundles."""
    return Pipeline._load_from_parts(manifest, toc, blobs, context)


def _capabilities(estimator):
    value = getattr(estimator, 'capabilities', {})
    return value() if callable(value) else value or {}


def _dispose_loaded(estimators):
    for estimator in reversed(estimators):
        try:
            if hasattr(estimator, 'dispose'):
                estimator.dispose()
        except Exception:
            # Preserve the load error; cleanup is best effort for partial state.
            pass


register(PIPELINE_TYPE_ID, _pipeline_loader, accepts_context=True)
