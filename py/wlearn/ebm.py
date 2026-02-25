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

        # Pre-convert cuts and scores to numpy arrays for fast predict
        self._cuts = []
        for f in model_data['features']:
            cuts = np.array(f.get('cuts', []), dtype=np.float64)
            self._cuts.append(cuts)

        self._terms = model_data['terms']
        self._term_scores = []
        for t in self._terms:
            self._term_scores.append(np.array(t['scores'], dtype=np.float64))

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


register('wlearn.ebm.classifier@1', EBMModel._from_bundle)
register('wlearn.ebm.regressor@1', EBMModel._from_bundle)
