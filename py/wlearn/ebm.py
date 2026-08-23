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
from ._capabilities import estimator_capabilities
from .bundle import encode_bundle, write_bundle_output
from .registry import register

MAX_C_INT = 2147483647


def _positive_c_int(value, name):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < 1 or value > MAX_C_INT):
        raise ValueError(f'{name} must be a positive int32 value')


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            np.isfinite(value))


def _validate_task_params(params, expected_task):
    if not isinstance(params, dict):
        raise ValueError('EBM params must be an object')
    for key in ('objective', 'task'):
        value = params.get(key)
        if value is not None and value != expected_task:
            raise ValueError(
                f'EBM {key} {value!r} does not match fitted '
                f'{expected_task} task')
    return params


def _validate_model_data(model_data, expected_task, n_classes):
    if (not isinstance(model_data, dict) or
            model_data.get('format') != 'ebm-json-v1' or
            model_data.get('task') != expected_task):
        raise ValueError(
            f'EBM model task/format does not match {expected_task}')

    for key in ('nFeatures', 'nTerms', 'nScores'):
        _positive_c_int(model_data.get(key), f'model {key}')
    n_features = model_data['nFeatures']
    n_terms = model_data['nTerms']
    n_scores = model_data['nScores']
    expected_scores = 1 if (
        expected_task == 'regression' or n_classes == 2) else n_classes
    if n_scores != expected_scores:
        raise ValueError('model nScores does not match task/classes')

    intercept = model_data.get('intercept')
    if (not isinstance(intercept, list) or len(intercept) != n_scores or
            not all(_finite_number(value) for value in intercept)):
        raise ValueError(
            'model intercept must contain one finite value per score')

    features = model_data.get('features')
    if not isinstance(features, list) or len(features) != n_features:
        raise ValueError('model features length does not match nFeatures')
    for index, feature in enumerate(features):
        if (not isinstance(feature, dict) or
                feature.get('type') not in ('continuous', 'nominal')):
            raise ValueError(f'model feature {index} has an invalid type')
        if feature['type'] == 'continuous':
            cuts = feature.get('cuts')
            if (not isinstance(cuts, list) or
                    not all(_finite_number(value) for value in cuts) or
                    any(cuts[i] <= cuts[i - 1]
                        for i in range(1, len(cuts))) or
                    len(cuts) > MAX_C_INT - 2):
                raise ValueError(
                    f'model feature {index} cuts must be finite and '
                    'strictly increasing')
        else:
            _positive_c_int(
                feature.get('nBins'), f'model feature {index} nBins')

    terms = model_data.get('terms')
    if not isinstance(terms, list) or len(terms) != n_terms:
        raise ValueError('model terms length does not match nTerms')
    for term_index, term in enumerate(terms):
        term_features = term.get('features') if isinstance(term, dict) else None
        bin_counts = term.get('binCounts') if isinstance(term, dict) else None
        if (not isinstance(term_features, list) or
                not 1 <= len(term_features) <= n_features or
                not isinstance(bin_counts, list) or
                len(bin_counts) != len(term_features)):
            raise ValueError(f'model term {term_index} has invalid dimensions')
        flat_size = 1
        seen_features = set()
        for dimension, (feature_index, bin_count) in enumerate(
                zip(term_features, bin_counts)):
            if (isinstance(feature_index, bool) or
                    not isinstance(feature_index, int) or
                    not 0 <= feature_index < n_features):
                raise ValueError(
                    f'model term {term_index} has an invalid feature index')
            if feature_index in seen_features:
                raise ValueError(
                    f'model term {term_index} has a duplicate feature index')
            seen_features.add(feature_index)
            _positive_c_int(
                bin_count,
                f'model term {term_index} binCounts[{dimension}]')
            feature = features[feature_index]
            if (feature['type'] == 'continuous' and
                    bin_count != len(feature['cuts']) + 2):
                raise ValueError(
                    f'model term {term_index} bin count disagrees with '
                    f'feature {feature_index}')
            if (feature['type'] == 'nominal' and
                    bin_count != feature['nBins']):
                raise ValueError(
                    f'model term {term_index} bin count disagrees with '
                    f'feature {feature_index}')
            if flat_size > MAX_C_INT // bin_count:
                raise ValueError(
                    f'model term {term_index} bin product exceeds int32 limits')
            flat_size *= bin_count
        if flat_size > MAX_C_INT // n_scores:
            raise ValueError(
                f'model term {term_index} score count exceeds int32 limits')
        score_count = flat_size * n_scores
        scores = term.get('scores')
        if (not isinstance(scores, list) or len(scores) != score_count or
                not all(_finite_number(value) for value in scores)):
            raise ValueError(
                f'model term {term_index} must contain exactly '
                f'{score_count} finite scores')

    return model_data


def _validate_matrix(X, expected_features=None):
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('X must be a rectangular numeric 2-D matrix') from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError('X must be a non-empty rectangular numeric 2-D matrix')
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f'X has {matrix.shape[1]} features; model expects '
            f'{expected_features}')
    if np.isinf(matrix).any():
        raise ValueError('X values must be finite or NaN')
    return matrix


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


def _find_feature_bins(feature, edges, vals, bin_count):
    if feature['type'] == 'continuous':
        return _find_bins(edges, vals)

    # Match the C/JS nominal path: integer-coded values select their bin;
    # missing or out-of-range values use the final term bin.
    bins = np.full(vals.shape, bin_count - 1, dtype=np.intp)
    finite_positions = np.flatnonzero(np.isfinite(vals))
    if finite_positions.size == 0:
        return bins
    finite_values = vals[finite_positions]
    valid = ((finite_values >= 0) & (finite_values < bin_count) &
             (finite_values == np.floor(finite_values)))
    positions = finite_positions[valid]
    bins[positions] = finite_values[valid].astype(np.intp)
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

        objective = self._params.get('objective')
        if objective not in (None, 'classification', 'regression'):
            raise ValueError(
                "objective must be 'classification' or 'regression'")
        task = self._params.get('task')
        if task not in (None, 'classification', 'regression'):
            raise ValueError(
                "task must be 'classification' or 'regression'")

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

        X = _validate_matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                f'y length ({y.size}) does not match X rows ({len(X)})')

        # Determine task
        if objective == 'regression':
            is_regressor = True
        elif objective == 'classification':
            is_regressor = False
        elif task == 'regression':
            is_regressor = True
        elif task == 'classification':
            is_regressor = False
        else:
            is_regressor = not np.all(y == np.floor(y.astype(np.float64)))

        if is_regressor:
            try:
                y_numeric = y.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError('Regression labels must be finite numbers') \
                    from error
            if not np.isfinite(y_numeric).all():
                raise ValueError('Regression labels must be finite numbers')
            y = y_numeric
        else:
            try:
                y_numeric = y.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError('Classifier labels must be int32 values') \
                    from error
            if (not np.isfinite(y_numeric).all() or
                    not np.equal(y_numeric, np.floor(y_numeric)).all() or
                    np.any(y_numeric < np.iinfo(np.int32).min) or
                    np.any(y_numeric > np.iinfo(np.int32).max)):
                raise ValueError('Classifier labels must be int32 values')
            if np.unique(y_numeric).size < 2:
                raise ValueError('Classification requires at least 2 classes')
            y = y_numeric.astype(np.int32)

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
        skip = {'objective', 'task'}
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
        type_id = manifest.get('typeId')
        if type_id not in {
                'wlearn.ebm.classifier@1',
                'wlearn.ebm.regressor@1'}:
            raise ValueError(f'Unsupported EBM bundle typeId: {type_id}')
        if (not isinstance(toc, list) or len(toc) != 1 or
                toc[0].get('id') != 'model' or
                toc[0].get('mediaType') != 'application/octet-stream'):
            raise ValueError(
                'EBM bundle must contain exactly one model artifact')
        entry = toc[0]

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        model_data = json.loads(blob.decode('utf-8'))

        params = manifest.get('params', {})
        metadata = manifest.get('metadata', {})
        expected_task = ('classification'
                         if type_id == 'wlearn.ebm.classifier@1'
                         else 'regression')
        _validate_task_params(params, expected_task)
        if expected_task == 'classification':
            classes = metadata.get('classes')
            n_classes = metadata.get('nClasses')
            if (isinstance(n_classes, bool) or
                    not isinstance(n_classes, int) or n_classes < 2 or
                    not isinstance(classes, list) or
                    len(classes) != n_classes or
                    any(isinstance(value, bool) or
                        not isinstance(value, int) or
                        value < np.iinfo(np.int32).min or
                        value > np.iinfo(np.int32).max
                        for value in classes) or
                    any(classes[index] <= classes[index - 1]
                        for index in range(1, len(classes)))):
                raise ValueError(f'{type_id} has invalid class metadata')
        elif (metadata.get('nClasses') != 0 or
              metadata.get('classes') not in (None, [])):
            raise ValueError(f'{type_id} regressor has classifier metadata')
        _validate_model_data(
            model_data, expected_task,
            metadata.get('nClasses', 0) if expected_task == 'classification'
            else 0)
        term_names = metadata.get('termNames')
        if (term_names is not None and
                (not isinstance(term_names, list) or
                 len(term_names) != model_data['nTerms'] or
                 not all(isinstance(value, str) for value in term_names))):
            raise ValueError(f'{type_id} has invalid termNames metadata')
        feature_names = metadata.get('featureNames')
        if (feature_names is not None and
                (not isinstance(feature_names, list) or
                 len(feature_names) != model_data['nFeatures'] or
                 not all(isinstance(value, str) for value in feature_names))):
            raise ValueError(f'{type_id} has invalid featureNames metadata')
        return EBMModel(model_data, params, metadata, raw_blob=blob)

    def _predict_scores(self, X):
        """Compute raw scores (before link function)."""
        X = _validate_matrix(X, self._n_features)
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
                bins = _find_feature_bins(
                    self._model_data['features'][fi], cuts, vals,
                    bin_counts[d])
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
        scores = self._predict_scores(X)
        n_samples = scores.shape[0]
        ns = self._n_scores

        if self._task == 'regression':
            return scores[:, 0]
        elif ns == 1:
            # Binary classification: sigmoid + threshold
            proba = 1.0 / (1.0 + np.exp(-scores[:, 0]))
            indexes = np.where(proba > 0.5, 1, 0)
        else:
            # Multiclass: argmax
            indexes = np.argmax(scores, axis=1)

        # Remap to original class labels
        if self._classes is None:
            raise ValueError('Classifier is missing class metadata')
        if np.any(indexes < 0) or np.any(indexes >= len(self._classes)):
            raise ValueError('Classifier produced an invalid class index')
        return self._classes[indexes].astype(np.int32, copy=False)

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
        X = _validate_matrix(X, self._n_features)
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
                bins = _find_feature_bins(
                    self._model_data['features'][fi], cuts, vals,
                    bin_counts[d])
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
        y = np.asarray(y)
        if y.size != preds.size:
            raise ValueError(
                f'y length ({y.size}) does not match predictions '
                f'({preds.size})')
        if self._task == 'regression':
            y = y.astype(np.float64)
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return float(np.mean(preds == y))

    def save(self, path=None):
        self._ensure_fitted()
        params = _validate_task_params(self.get_params(), self._task)
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

        bundle = encode_bundle(
            {'typeId': type_id, 'params': params,
             'metadata': metadata},
            [{'id': 'model', 'data': json_bytes}],
        )
        return write_bundle_output(bundle, path)

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
    def classes(self):
        self._ensure_fitted()
        return None if self._classes is None else self._classes.copy()

    @property
    def capabilities(self):
        task = self._task or self._params.get('objective')
        classifier = task == 'classification'
        regressor = task == 'regression'
        return estimator_capabilities(
            classifier=classifier,
            regressor=regressor,
            predict_proba=classifier,
            featureImportances=True,
        )

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
            'minSamplesLeaf': {'type': 'int_uniform', 'low': 1, 'high': 10},
        }


register('wlearn.ebm.classifier@1', EBMModel._from_bundle)
register('wlearn.ebm.regressor@1', EBMModel._from_bundle)
