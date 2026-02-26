"""ML preprocessing: imputation, encoding, scaling.

Learns parameters during fit(), applies during transform().
State is serializable for .wlrn bundles.
"""

import math
import json

import numpy as np

from .errors import ValidationError


class Preprocessor:
    """Automatic feature preprocessing for tabular data.

    Handles missing values, categorical encoding, and feature scaling.
    Learns parameters from training data and applies them consistently
    at inference time.
    """

    def __init__(self, impute='auto', encode='auto', scale=False,
                 max_categories=20):
        self._config = {
            'impute': impute,
            'encode': encode,
            'scale': scale,
            'maxCategories': max_categories,
        }
        self._fitted = False
        self._col_types = None
        self._impute_values = None
        self._encodings = None
        self._scale_params = None
        self._output_cols = 0

    def fit(self, X, y=None):
        """Learn preprocessing parameters from training data."""
        X = np.asarray(X, dtype=np.float64)
        rows, cols = X.shape

        self._col_types = [None] * cols
        self._impute_values = np.zeros(cols, dtype=np.float64)
        col_stats = [None] * cols

        for c in range(cols):
            col = X[:, c]
            mask = ~np.isnan(col)
            values = col[mask]
            has_nan = not mask.all()

            unique = set(values.tolist())
            is_integer = all(v == int(v) for v in values)
            is_categorical = (
                self._config['encode'] is not False and
                is_integer and
                len(unique) <= self._config['maxCategories'] and
                len(unique) >= 2
            )

            self._col_types[c] = 'categorical' if is_categorical else 'numeric'

            if self._config['impute'] is not False and has_nan:
                if is_categorical:
                    # Mode
                    from collections import Counter
                    counter = Counter(values.tolist())
                    self._impute_values[c] = counter.most_common(1)[0][0]
                else:
                    # Mean
                    self._impute_values[c] = values.mean() if len(values) > 0 else 0.0

            col_stats[c] = {
                'values': values,
                'unique': unique,
                'sum': float(values.sum()) if len(values) > 0 else 0.0,
                'count': len(values),
            }

        # Build encodings
        self._encodings = [None] * cols
        output_idx = 0

        if self._config['encode'] is not False:
            for c in range(cols):
                if self._col_types[c] == 'categorical':
                    sorted_vals = sorted(col_stats[c]['unique'])
                    if self._config['encode'] == 'label':
                        mapping = {v: i for i, v in enumerate(sorted_vals)}
                        self._encodings[c] = {
                            'type': 'label',
                            'mapping': mapping,
                            'startIdx': output_idx,
                        }
                        output_idx += 1
                    else:
                        mapping = {v: output_idx + i
                                   for i, v in enumerate(sorted_vals)}
                        self._encodings[c] = {
                            'type': 'onehot',
                            'mapping': mapping,
                            'startIdx': output_idx,
                            'size': len(sorted_vals),
                        }
                        output_idx += len(sorted_vals)
                else:
                    output_idx += 1
        else:
            output_idx = cols

        self._output_cols = output_idx

        # Compute scaling parameters
        self._scale_params = [None] * self._output_cols

        if self._config['scale']:
            out_c = 0
            for c in range(cols):
                if (self._col_types[c] == 'categorical' and
                        self._config['encode'] is not False):
                    enc = self._encodings[c]
                    out_c += enc['size'] if enc['type'] == 'onehot' else 1
                    continue
                vals = col_stats[c]['values']
                if self._config['scale'] == 'standard':
                    mean = float(vals.mean()) if len(vals) > 0 else 0.0
                    std = float(vals.std()) if len(vals) > 0 else 1.0
                    self._scale_params[out_c] = {
                        'mean': mean,
                        'std': std if std > 0 else 1.0,
                    }
                elif self._config['scale'] == 'minmax':
                    mn = float(vals.min()) if len(vals) > 0 else 0.0
                    mx = float(vals.max()) if len(vals) > 0 else 1.0
                    rng = mx - mn
                    self._scale_params[out_c] = {
                        'min': mn,
                        'range': rng if rng > 0 else 1.0,
                    }
                out_c += 1

        self._fitted = True
        return self

    def transform(self, X):
        """Apply learned preprocessing to new data."""
        if not self._fitted:
            raise ValidationError('Preprocessor not fitted')
        X = np.asarray(X, dtype=np.float64)
        rows, cols = X.shape
        out_cols = self._output_cols
        out = np.zeros((rows, out_cols), dtype=np.float64)

        for r in range(rows):
            out_c = 0
            for c in range(cols):
                v = X[r, c]

                # Impute missing
                if math.isnan(v) and self._config['impute'] is not False:
                    v = self._impute_values[c]

                # Encode
                if (self._col_types[c] == 'categorical' and
                        self._encodings[c] is not None):
                    enc = self._encodings[c]
                    if enc['type'] == 'onehot':
                        idx = enc['mapping'].get(v)
                        if idx is not None:
                            out[r, idx] = 1.0
                        out_c = enc['startIdx'] + enc['size']
                    else:
                        label = enc['mapping'].get(v, -1)
                        out[r, out_c] = label
                        out_c += 1
                else:
                    out[r, out_c] = v
                    out_c += 1

        # Apply scaling
        if self._config['scale']:
            for j in range(out_cols):
                sp = self._scale_params[j]
                if sp is None:
                    continue
                if self._config['scale'] == 'standard':
                    out[:, j] = (out[:, j] - sp['mean']) / sp['std']
                elif self._config['scale'] == 'minmax':
                    out[:, j] = (out[:, j] - sp['min']) / sp['range']

        return out

    def fit_transform(self, X, y=None):
        """Fit and transform in one call."""
        self.fit(X, y)
        return self.transform(X)

    def get_state(self):
        """Serialize fitted state."""
        if not self._fitted:
            raise ValidationError('Preprocessor not fitted')
        return {
            'config': self._config,
            'colTypes': self._col_types,
            'imputeValues': self._impute_values.tolist(),
            'encodings': [
                {
                    'type': e['type'],
                    'mapping': list(e['mapping'].items()),
                    'startIdx': e['startIdx'],
                    'size': e.get('size'),
                } if e else None
                for e in self._encodings
            ],
            'scaleParams': self._scale_params,
            'outputCols': self._output_cols,
        }

    @classmethod
    def from_state(cls, state):
        """Restore from serialized state."""
        pp = cls(**{
            'impute': state['config']['impute'],
            'encode': state['config']['encode'],
            'scale': state['config']['scale'],
            'max_categories': state['config']['maxCategories'],
        })
        pp._col_types = state['colTypes']
        pp._impute_values = np.array(state['imputeValues'], dtype=np.float64)
        pp._encodings = [
            {
                'type': e['type'],
                'mapping': {k: v for k, v in e['mapping']},
                'startIdx': e['startIdx'],
                'size': e.get('size'),
            } if e else None
            for e in state['encodings']
        ]
        pp._scale_params = state['scaleParams']
        pp._output_cols = state['outputCols']
        pp._fitted = True
        return pp

    @property
    def is_fitted(self):
        return self._fitted

    @property
    def output_cols(self):
        return self._output_cols

    def dispose(self):
        pass
