"""Python wrapper for @wlearn/ebm bundles.

Loads WLRN bundles produced by JS @wlearn/ebm, predicts using pure numpy
lookup-table evaluation, and saves back to WLRN bundles that JS can load.

Blob format: UTF-8 JSON string with structure:
  {
    "format": "ebm-json-v1",
    "task": "classification" | "regression",
    "nFeatures": int,
    "nTerms": int,
    "nScores": int,
    "intercept": [float, ...],
    "features": [{"type": "continuous", "cuts": [float, ...]}, ...],
    "terms": [{"features": [int], "binCounts": [int], "scores": [float]}, ...]
  }
"""

import json

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register


def _find_bins(edges, vals):
    """Find bin indices matching the C find_bin() behavior.

    C uses: if (val < edges[mid]) hi = mid; else lo = mid + 1;
    This is equivalent to np.searchsorted(side='right').

    Bins:
      0            : val < edges[0]
      i (1..n-1)   : edges[i-1] <= val < edges[i]
      n_cuts       : val >= edges[-1] (also used for NaN)
    """
    bins = np.searchsorted(edges, vals, side='right')
    nan_mask = np.isnan(vals)
    if np.any(nan_mask):
        bins[nan_mask] = len(edges)
    return bins


class EBMModel:
    def __init__(self, model_data, params, metadata, raw_blob=None):
        self._model_data = model_data
        self._params = dict(params)
        self._disposed = False
        self._fitted = True
        self._raw_blob = raw_blob  # original blob bytes for round-trip identity

        self._task = model_data['task']
        self._n_features = model_data['nFeatures']
        self._n_terms = model_data['nTerms']
        self._n_scores = model_data['nScores']
        self._intercept = np.array(model_data['intercept'], dtype=np.float64)

        self._n_classes = metadata.get('nClasses', 0)
        classes = metadata.get('classes')
        self._classes = np.array(classes, dtype=np.int32) if classes else None
        self._term_names = metadata.get('termNames')
        self._feature_names = metadata.get('featureNames')

        self._setup_arrays(model_data)

    def _setup_arrays(self, model_data):
        """Pre-convert cuts and scores to numpy arrays for fast predict."""
        self._cuts = []
        for f in model_data['features']:
            cuts = np.array(f.get('cuts', []), dtype=np.float64)
            self._cuts.append(cuts)

        self._terms = model_data['terms']
        self._term_scores = []
        for t in self._terms:
            self._term_scores.append(np.array(t['scores'], dtype=np.float64))

    @classmethod
    def create(cls, params=None):
        """Create an unfitted EBM model."""
        obj = cls.__new__(cls)
        obj._model_data = None
        obj._params = dict(params) if params else {}
        obj._disposed = False
        obj._fitted = False
        obj._raw_blob = None
        obj._task = None
        obj._n_features = 0
        obj._n_terms = 0
        obj._n_scores = 0
        obj._intercept = None
        obj._n_classes = 0
        obj._classes = None
        obj._term_names = None
        obj._feature_names = None
        obj._cuts = []
        obj._terms = []
        obj._term_scores = []
        return obj

    def fit(self, X, y):
        """Train an EBM model using the interpret package.

        Requires the ``interpret`` package (``pip install interpret``).
        Predict/save/load only need numpy.
        """
        if self._disposed:
            raise DisposedError('EBMModel has been disposed.')

        try:
            from interpret.glassbox import (
                ExplainableBoostingClassifier,
                ExplainableBoostingRegressor,
            )
        except ImportError:
            raise ImportError(
                'interpret package required for fit(). '
                'Install with: pip install interpret'
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Determine task
        objective = self._params.get('objective')
        if objective == 'regression':
            is_regressor = True
        elif objective == 'classification':
            is_regressor = False
        else:
            is_regressor = not np.all(y == np.floor(y.astype(np.float64)))

        interpret_params = self._map_params(self._params)

        if is_regressor:
            ebm = ExplainableBoostingRegressor(**interpret_params)
            ebm.fit(X, y)
            self._task = 'regression'
            self._n_classes = 0
            self._classes = None
        else:
            ebm = ExplainableBoostingClassifier(**interpret_params)
            ebm.fit(X, y)
            self._task = 'classification'
            classes = ebm.classes_.astype(np.int32)
            self._classes = classes
            self._n_classes = len(classes)

        self._model_data = self._extract_model_data(ebm, is_regressor)
        self._raw_blob = None

        self._n_features = self._model_data['nFeatures']
        self._n_terms = self._model_data['nTerms']
        self._n_scores = self._model_data['nScores']
        self._intercept = np.array(self._model_data['intercept'], dtype=np.float64)
        self._setup_arrays(self._model_data)

        if hasattr(ebm, 'term_names_'):
            self._term_names = [str(n) for n in ebm.term_names_]
        if hasattr(ebm, 'feature_names_in_'):
            self._feature_names = [str(n) for n in ebm.feature_names_in_]

        self._fitted = True
        return self

    @staticmethod
    def _map_params(params):
        """Map wlearn camelCase params to interpret snake_case."""
        mapping = {
            'learningRate': 'learning_rate',
            'maxRounds': 'max_rounds',
            'earlyStoppingRounds': 'early_stopping_rounds',
            'maxLeaves': 'max_leaves',
            'minSamplesLeaf': 'min_samples_leaf',
            'maxInteractions': 'interactions',
            'maxBins': 'max_bins',
            'outerBags': 'outer_bags',
            'innerBags': 'inner_bags',
            'seed': 'random_state',
        }
        skip = {'objective'}
        out = {}
        for k, v in params.items():
            if k in skip:
                continue
            out[mapping.get(k, k)] = v

        # Match JS behavior: same bins for interactions and mains
        if 'max_interaction_bins' not in out:
            out['max_interaction_bins'] = out.get('max_bins', 256)

        return out

    @staticmethod
    def _extract_model_data(ebm, is_regressor):
        """Convert interpret fitted model to ebm-json-v1 format."""
        n_features = len(ebm.feature_names_in_)
        n_terms = len(ebm.term_features_)

        if is_regressor:
            n_scores = 1
        else:
            n_classes = len(ebm.classes_)
            n_scores = 1 if n_classes <= 2 else n_classes

        # Intercept
        intercept_raw = np.atleast_1d(np.asarray(ebm.intercept_, dtype=np.float64))
        intercept = [float(v) for v in intercept_raw[:n_scores]]

        # Features (bin edges)
        features = []
        for fi in range(n_features):
            ftype = ebm.feature_types_in_[fi]
            if ftype == 'continuous':
                bins_fi = ebm.bins_[fi]
                # bins_ is list-of-arrays (one per resolution level)
                if isinstance(bins_fi, list):
                    cuts_arr = np.asarray(bins_fi[0], dtype=np.float64)
                else:
                    cuts_arr = np.asarray(bins_fi, dtype=np.float64)
                features.append({
                    'type': 'continuous',
                    'cuts': [float(c) for c in cuts_arr],
                })
            else:
                # Nominal: bins_ is a dict mapping categories to bin indices
                bins_fi = ebm.bins_[fi]
                if isinstance(bins_fi, list):
                    bins_fi = bins_fi[0]
                n_bins = len(bins_fi) if isinstance(bins_fi, dict) else int(bins_fi)
                features.append({
                    'type': 'nominal',
                    'nBins': n_bins,
                })

        # Terms: strip the unseen bin (last index) from each spatial dimension
        terms = []
        for t in range(n_terms):
            term_features = [int(f) for f in ebm.term_features_[t]]
            scores_raw = np.asarray(ebm.term_scores_[t], dtype=np.float64)
            n_dims = len(term_features)

            # Strip unseen bin from each spatial dimension
            slices = tuple(slice(0, -1) for _ in range(n_dims))
            if scores_raw.ndim > n_dims:
                # Multiclass: extra axis for classes
                slices = slices + (slice(None),)
            scores_stripped = scores_raw[slices]

            bin_counts = [int(scores_stripped.shape[d]) for d in range(n_dims)]
            flat_scores = [float(s) for s in scores_stripped.ravel()]

            terms.append({
                'features': term_features,
                'binCounts': bin_counts,
                'scores': flat_scores,
            })

        return {
            'format': 'ebm-json-v1',
            'task': 'regression' if is_regressor else 'classification',
            'nFeatures': n_features,
            'nTerms': n_terms,
            'nScores': n_scores,
            'intercept': intercept,
            'features': features,
            'terms': terms,
        }

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        model_data = json.loads(blob.decode('utf-8'))

        params = manifest.get('params', {})
        metadata = manifest.get('metadata', {})
        return EBMModel(model_data, params, metadata, raw_blob=blob)

    def _predict_scores(self, X):
        """Compute raw scores (before link function)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples, n_features = X.shape

        ns = self._n_scores
        out = np.tile(self._intercept, (n_samples, 1))  # (n_samples, n_scores)

        for t_idx, term in enumerate(self._terms):
            features = term['features']
            bin_counts = term['binCounts']
            n_dims = len(features)
            scores = self._term_scores[t_idx]

            # Compute bin index for each dimension
            dim_bins = []
            for d in range(n_dims):
                fi = features[d]
                cuts = self._cuts[fi]
                vals = X[:, fi]
                bins = _find_bins(cuts, vals)
                # Clamp to valid range
                np.clip(bins, 0, bin_counts[d] - 1, out=bins)
                dim_bins.append(bins)

            # Compute flat index (row-major, same as C: last dim varies fastest)
            flat_idx = np.zeros(n_samples, dtype=np.intp)
            stride = 1
            for d in range(n_dims - 1, -1, -1):
                flat_idx += dim_bins[d] * stride
                stride *= bin_counts[d]

            # Look up scores and add to output
            if ns == 1:
                out[:, 0] += scores[flat_idx]
            else:
                for s in range(ns):
                    out[:, s] += scores[flat_idx * ns + s]

        return out

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = self._predict_scores(X)
        n_samples = scores.shape[0]
        ns = self._n_scores

        if self._task == 'regression':
            preds = scores[:, 0]
        elif ns == 1:
            # Binary classification: sigmoid + threshold
            proba = 1.0 / (1.0 + np.exp(-scores[:, 0]))
            preds = np.where(proba > 0.5, 1.0, 0.0)
        else:
            # Multiclass: argmax
            preds = np.argmax(scores, axis=1).astype(np.float64)

        # Remap to original class labels
        if self._task != 'regression' and self._classes is not None:
            for i in range(n_samples):
                idx = int(round(preds[i]))
                if 0 <= idx < len(self._classes):
                    preds[i] = float(self._classes[idx])

        return preds

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task == 'regression':
            raise ValueError('predict_proba only for classification')

        scores = self._predict_scores(X)
        ns = self._n_scores

        if ns == 1:
            # Binary: return [P(0), P(1)] per sample
            p1 = 1.0 / (1.0 + np.exp(-scores[:, 0]))
            proba = np.column_stack([1.0 - p1, p1])
        else:
            # Multiclass: softmax
            max_scores = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return proba.ravel()

    def explain(self, X):
        """Per-term additive contributions for each sample."""
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        ns = self._n_scores
        nt = self._n_terms

        contributions = np.zeros((n_samples, nt, ns), dtype=np.float64)

        for t_idx, term in enumerate(self._terms):
            features = term['features']
            bin_counts = term['binCounts']
            n_dims = len(features)
            scores = self._term_scores[t_idx]

            dim_bins = []
            for d in range(n_dims):
                fi = features[d]
                cuts = self._cuts[fi]
                vals = X[:, fi]
                bins = _find_bins(cuts, vals)
                np.clip(bins, 0, bin_counts[d] - 1, out=bins)
                dim_bins.append(bins)

            flat_idx = np.zeros(n_samples, dtype=np.intp)
            stride = 1
            for d in range(n_dims - 1, -1, -1):
                flat_idx += dim_bins[d] * stride
                stride *= bin_counts[d]

            if ns == 1:
                contributions[:, t_idx, 0] = scores[flat_idx]
            else:
                for s in range(ns):
                    contributions[:, t_idx, s] = scores[flat_idx * ns + s]

        return {
            'intercept': self._intercept.tolist(),
            'contributions': contributions.ravel(),
            'termNames': list(self._term_names) if self._term_names else [],
            'nTerms': nt,
            'nSamples': n_samples,
            'nScores': ns,
        }

    def feature_importances(self):
        """Mean absolute score per term."""
        self._ensure_fitted()
        importances = np.zeros(self._n_terms, dtype=np.float64)
        for t in range(self._n_terms):
            scores = self._term_scores[t]
            importances[t] = np.mean(np.abs(scores))
        return importances

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if self._task == 'regression':
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()
        # Reuse original blob bytes for round-trip identity
        if self._raw_blob is not None:
            json_bytes = self._raw_blob
        else:
            json_str = json.dumps(self._model_data, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')

        type_id = ('wlearn.ebm.regressor@1'
                   if self._task == 'regression'
                   else 'wlearn.ebm.classifier@1')

        metadata = {
            'nClasses': int(self._n_classes),
            'classes': self._classes.tolist() if self._classes is not None else [],
        }
        if self._term_names is not None:
            metadata['termNames'] = self._term_names
        if self._feature_names is not None:
            metadata['featureNames'] = self._feature_names

        return encode_bundle(
            {'typeId': type_id, 'params': self.get_params(),
             'metadata': metadata},
            [{'id': 'model', 'data': json_bytes}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._fitted = False
        self._model_data = None
        self._cuts = None
        self._term_scores = None

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('EBMModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('EBMModel is not fitted.')


    @classmethod
    def default_search_space(cls):
        return {
            'learningRate': {'type': 'log_uniform', 'low': 0.001, 'high': 0.1},
            'maxRounds': {'type': 'int_uniform', 'low': 1000, 'high': 10000},
            'maxLeaves': {'type': 'int_uniform', 'low': 2, 'high': 5},
            'maxInteractions': {'type': 'int_uniform', 'low': 0, 'high': 20},
            'maxBins': {'type': 'categorical', 'values': [128, 256, 512]},
            'outerBags': {'type': 'int_uniform', 'low': 4, 'high': 16},
            'minSamplesLeaf': {'type': 'int_uniform', 'low': 1, 'high': 10},
        }


register('wlearn.ebm.classifier@1', EBMModel._from_bundle)
register('wlearn.ebm.regressor@1', EBMModel._from_bundle)
