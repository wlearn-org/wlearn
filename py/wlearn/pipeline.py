from .errors import ValidationError, NotFittedError, DisposedError
from .bundle import encode_bundle, decode_bundle
from .registry import register, load as registry_load

PIPELINE_TYPE_ID = 'wlearn.pipeline@1'


class Pipeline:
    """Pipeline estimator: chains transformer steps + final estimator.

    Supports fit/predict/score with transformer + estimator chains,
    as well as save/load from WLRN bundles.
    """

    def __init__(self, steps):
        """Create a pipeline from (name, estimator) pairs.

        Args:
            steps: list of (name, estimator) tuples
        """
        if not steps:
            raise ValidationError('Pipeline requires at least one step')
        self._steps = list(steps)
        self._fitted = False
        self._disposed = False

    def fit(self, X, y):
        """Fit the pipeline: transform through intermediates, fit last step.

        For intermediate steps that have a transform method:
        - If fit_transform exists, use it (avoids fitting + transforming separately)
        - Otherwise, call fit() then transform()
        The last step is only fitted (not transformed).
        """
        self._ensure_alive()
        current = X
        for i, (_, est) in enumerate(self._steps[:-1]):
            if hasattr(est, 'fit_transform'):
                current = est.fit_transform(current, y)
            else:
                est.fit(current, y)
                current = est.transform(current)
        # Last step: fit only
        _, last = self._steps[-1]
        last.fit(current, y)
        self._fitted = True
        return self

    def _transform_through(self, X):
        """Transform X through all intermediate steps (not the last)."""
        current = X
        for _, est in self._steps[:-1]:
            current = est.transform(current)
        return current

    def predict(self, X):
        """Transform through intermediates, predict with last step."""
        self._ensure_fitted()
        transformed = self._transform_through(X)
        _, last = self._steps[-1]
        return last.predict(transformed)

    def predict_proba(self, X):
        """Transform through intermediates, predict_proba with last step."""
        self._ensure_fitted()
        _, last = self._steps[-1]
        if not hasattr(last, 'predict_proba'):
            raise ValidationError('Last step does not support predict_proba')
        transformed = self._transform_through(X)
        return last.predict_proba(transformed)

    def score(self, X, y):
        """Transform through intermediates, score with last step."""
        self._ensure_fitted()
        transformed = self._transform_through(X)
        _, last = self._steps[-1]
        return last.score(transformed, y)

    @classmethod
    def load(cls, data):
        """Load a pipeline from a WLRN bundle.

        Each step's artifact is loaded via the global registry.

        Args:
            data: bytes (WLRN bundle)

        Returns:
            Pipeline instance (fitted)
        """
        manifest, toc, blobs = decode_bundle(data)
        steps = []
        for step_info in manifest.get('steps', []):
            name = step_info['name']
            entry = next((t for t in toc if t['id'] == name), None)
            if entry is None:
                raise ValidationError(
                    f'No artifact found for pipeline step "{name}"')
            blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            estimator = registry_load(blob)
            steps.append((name, estimator))
        pipe = cls(steps)
        pipe._fitted = True
        return pipe

    def save(self):
        """Save pipeline to a WLRN bundle.

        Returns:
            bytes
        """
        self._ensure_fitted()
        manifest = {
            'typeId': PIPELINE_TYPE_ID,
            'steps': [
                {'name': name, 'params': est.get_params()
                 if hasattr(est, 'get_params') else {}}
                for name, est in self._steps
            ],
        }
        artifacts = [
            {'id': name, 'data': est.save(), 'mediaType': 'application/x-wlearn-bundle'}
            for name, est in self._steps
        ]
        return encode_bundle(manifest, artifacts)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        for _, est in self._steps:
            if hasattr(est, 'dispose'):
                est.dispose()

    def get_params(self):
        return {
            name: est.get_params() if hasattr(est, 'get_params') else {}
            for name, est in self._steps
        }

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('Pipeline has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('Pipeline is not fitted. Call fit() first.')


def _pipeline_loader(manifest, toc, blobs):
    """Registry loader for wlearn.pipeline@1 bundles."""
    steps = []
    for step_info in manifest.get('steps', []):
        name = step_info['name']
        entry = next((t for t in toc if t['id'] == name), None)
        if entry is None:
            raise ValidationError(
                f'No artifact found for pipeline step "{name}"')
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        estimator = registry_load(blob)
        steps.append((name, estimator))
    pipe = Pipeline(steps)
    pipe._fitted = True
    return pipe


register(PIPELINE_TYPE_ID, _pipeline_loader)
