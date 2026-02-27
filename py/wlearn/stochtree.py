"""Python wrapper for @wlearn/stochtree bundles.

Loads WLRN bundles produced by JS @wlearn/stochtree, predicts using pure numpy
tree traversal (no stochtree C++ dependency needed for inference), trains using
the upstream stochtree Python package (pip install stochtree), and saves back to
WLRN bundles that JS can load.

Blob format: UTF-8 JSON string with stochtree ForestContainer + metadata.
"""

import json
import math

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register


def _traverse_tree(tree, X):
    """Traverse a stochtree tree for all samples, return leaf values.

    The tree JSON has:
      split_index: feature index per node (-1 for leaf)
      threshold: split value per node
      left/right: child node indices (-1 unused)
      node_type: 1=internal, 0=leaf
      leaf_value: values at each node (only leaf nodes meaningful)

    Split convention: go left if value <= threshold (matches upstream
    SplitTrueNumeric in stochtree/tree.h).

    Returns array of shape (n_samples,).
    """
    n_samples = X.shape[0]
    node_type = tree['node_type']
    split_index = tree['split_index']
    threshold = tree['threshold']
    left = tree['left']
    right = tree['right']
    leaf_value = tree['leaf_value']

    result = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        node = 0
        while node_type[node] == 1:  # 1 = internal node
            if X[i, split_index[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        result[i] = leaf_value[node]
    return result


def _predict_forest(forest_json, X):
    """Predict from a single forest (sum of tree predictions)."""
    n_samples = X.shape[0]
    num_trees = forest_json['num_trees']
    result = np.zeros(n_samples, dtype=np.float64)
    for t in range(num_trees):
        tree = forest_json[f'tree_{t}']
        result += _traverse_tree(tree, X)
    return result


def _upstream_to_wasm_json(upstream_json, task, num_features, params,
                           classes=None):
    """Convert upstream stochtree BARTModel JSON to WASM bundle JSON format.

    Upstream format (from stochtree.BARTModel.to_json()):
      - forests.forest_0: ForestContainer dict
      - outcome_mean / outcome_scale: standardization params
      - parameters.sigma2_global_samples: variance samples
      - probit_outcome_model: bool

    WASM bundle format:
      - forest_container: ForestContainer dict
      - y_bar / y_std: standardization params
      - sigma2_samples: variance samples
      - task: 0=regression, 1=classification
    """
    fc = upstream_json['forests']['forest_0']
    sigma2 = upstream_json.get('parameters', {}).get(
        'sigma2_global_samples', [])

    return {
        'task': task,
        'y_bar': upstream_json.get('outcome_mean', 0.0),
        'y_std': upstream_json.get('outcome_scale', 1.0),
        'num_features': num_features,
        'num_trees': params.get('numTrees', fc.get('num_trees', 200)),
        'num_gfr': params.get('numGfr', 5),
        'num_burnin': params.get('numBurnin', 0),
        'num_samples': fc.get('num_samples', 0),
        'alpha': params.get('alpha', 0.95),
        'beta': params.get('beta', 2.0),
        'min_samples_leaf': params.get('minSamplesLeaf', 5),
        'max_depth': params.get('maxDepth', -1),
        'cutpoint_grid': params.get('cutpointGrid', 100),
        'leaf_scale': params.get('leafScale', -1.0),
        'random_seed': params.get('seed', 42),
        'nr_class': 2 if task == 1 else 0,
        'classes': list(classes) if classes is not None else [],
        'sigma2_samples': sigma2,
        'forest_container': fc,
    }


class BARTModel:
    def __init__(self, model_json, params, metadata, raw_blob=None):
        self._model_json = model_json
        self._params = dict(params)
        self._disposed = False
        self._fitted = True
        self._raw_blob = raw_blob

        self._task = model_json.get('task', 0)  # 0=reg, 1=clf
        self._y_bar = model_json.get('y_bar', 0.0)
        self._y_std = model_json.get('y_std', 1.0)
        self._num_features = model_json.get('num_features', 0)
        self._nr_class = metadata.get('nrClass', 0)
        classes = metadata.get('classes')
        self._classes = np.array(classes, dtype=np.int32) if classes else None

        # Parse forest container
        fc = model_json.get('forest_container', {})
        self._num_samples = fc.get('num_samples', 0)
        self._forests = []
        for s in range(self._num_samples):
            key = f'forest_{s}'
            if key in fc:
                self._forests.append(fc[key])

        self._sigma2_samples = np.array(
            model_json.get('sigma2_samples', []), dtype=np.float64)

    @classmethod
    def create(cls, params=None):
        """Create an unfitted BARTModel."""
        obj = cls.__new__(cls)
        obj._model_json = None
        obj._params = dict(params) if params else {}
        obj._disposed = False
        obj._fitted = False
        obj._raw_blob = None
        obj._task = 0
        obj._y_bar = 0.0
        obj._y_std = 1.0
        obj._num_features = 0
        obj._nr_class = 0
        obj._classes = None
        obj._num_samples = 0
        obj._forests = []
        obj._sigma2_samples = np.array([], dtype=np.float64)
        return obj

    def fit(self, X, y):
        """Train a BART model using the upstream stochtree package.

        Requires the ``stochtree`` package (``pip install stochtree``).
        Predict/save/load only need numpy.
        """
        if self._disposed:
            raise DisposedError('BARTModel has been disposed.')

        try:
            from stochtree import BARTModel as UpstreamBARTModel
        except ImportError:
            raise ImportError(
                'stochtree package required for fit(). '
                'Install with: pip install stochtree'
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        num_features = X.shape[1]

        # Determine task
        objective = self._params.get('objective')
        if objective == 'regression':
            is_classifier = False
        elif objective == 'classification':
            is_classifier = True
        else:
            unique_y = np.unique(y)
            is_classifier = (len(unique_y) <= 2
                             and np.all(unique_y == np.floor(unique_y)))

        # Map params
        num_trees = self._params.get('numTrees', 200)
        num_gfr = self._params.get('numGfr', 5)
        num_burnin = self._params.get('numBurnin', 0)
        num_samples = self._params.get('numSamples', 100)
        alpha = self._params.get('alpha', 0.95)
        beta = self._params.get('beta', 2.0)
        min_samples_leaf = self._params.get('minSamplesLeaf', 5)
        max_depth = self._params.get('maxDepth', -1)
        cutpoint_grid = self._params.get('cutpointGrid', 100)
        seed = self._params.get('seed', 42)

        general_params = {
            'random_seed': seed,
            'standardize': True,
            'cutpoint_grid_size': cutpoint_grid,
        }

        mean_forest_params = {
            'num_trees': num_trees,
            'alpha': alpha,
            'beta': beta,
            'min_samples_leaf': min_samples_leaf,
        }
        if max_depth > 0:
            mean_forest_params['max_depth'] = max_depth

        if is_classifier:
            # Probit BART classification
            general_params['probit_outcome_model'] = True
            general_params['sample_sigma2_global'] = False

            classes = np.unique(y).astype(np.int32)
            # Remap y to 0/1
            y_binary = np.zeros(len(y), dtype=np.float64)
            y_binary[y == classes[1]] = 1.0
        else:
            classes = None
            y_binary = y

        model = UpstreamBARTModel()
        model.sample(
            X, y_binary,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_samples,
            general_params=general_params,
            mean_forest_params=mean_forest_params,
        )

        # Convert upstream format to WASM bundle format
        upstream_json = json.loads(model.to_json())
        task = 1 if is_classifier else 0
        classes_list = classes.tolist() if classes is not None else []

        wasm_json = _upstream_to_wasm_json(
            upstream_json, task, num_features, self._params,
            classes=classes_list)

        # Re-init from the converted JSON
        self._model_json = wasm_json
        self._raw_blob = None
        self._task = task
        self._y_bar = wasm_json['y_bar']
        self._y_std = wasm_json['y_std']
        self._num_features = num_features
        self._nr_class = 2 if is_classifier else 0
        self._classes = classes

        fc = wasm_json['forest_container']
        self._num_samples = fc.get('num_samples', 0)
        self._forests = []
        for s in range(self._num_samples):
            key = f'forest_{s}'
            if key in fc:
                self._forests.append(fc[key])

        self._sigma2_samples = np.array(
            wasm_json.get('sigma2_samples', []), dtype=np.float64)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        model_json = json.loads(blob.decode('utf-8'))

        params = manifest.get('params', {})
        metadata = manifest.get('metadata', {})
        return BARTModel(model_json, params, metadata, raw_blob=blob)

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        # Average predictions across posterior forest samples
        avg = np.zeros(n_samples, dtype=np.float64)
        for forest in self._forests:
            avg += _predict_forest(forest, X)
        avg /= len(self._forests)

        if self._task == 0:
            # Regression: de-standardize
            return avg * self._y_std + self._y_bar
        else:
            # Classification: probit link -> class labels
            probs = np.array([
                0.5 * math.erfc(-v / math.sqrt(2.0)) for v in avg
            ])
            result = np.empty(n_samples, dtype=np.float64)
            for i in range(n_samples):
                result[i] = (float(self._classes[1]) if probs[i] >= 0.5
                             else float(self._classes[0]))
            return result

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task == 0:
            raise ValueError('predict_proba only for classification')

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        avg = np.zeros(n_samples, dtype=np.float64)
        for forest in self._forests:
            avg += _predict_forest(forest, X)
        avg /= len(self._forests)

        probs = np.array([
            0.5 * math.erfc(-v / math.sqrt(2.0)) for v in avg
        ])
        result = np.empty(n_samples * 2, dtype=np.float64)
        for i in range(n_samples):
            result[i * 2] = 1.0 - probs[i]
            result[i * 2 + 1] = probs[i]
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if self._task == 0:
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()
        if self._raw_blob is not None:
            json_bytes = self._raw_blob
        else:
            json_str = json.dumps(self._model_json, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')

        type_id = ('wlearn.stochtree.regressor@1'
                   if self._task == 0
                   else 'wlearn.stochtree.classifier@1')

        return encode_bundle(
            {
                'typeId': type_id,
                'params': self.get_params(),
                'metadata': {
                    'nrClass': int(self._nr_class),
                    'classes': (self._classes.tolist()
                                if self._classes is not None else []),
                    'numFeatures': int(self._num_features),
                },
            },
            [{'id': 'model', 'data': json_bytes}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._fitted = False
        self._model_json = None
        self._forests = None

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
            raise DisposedError('BARTModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('BARTModel is not fitted.')

    @classmethod
    def default_search_space(cls):
        return {
            'numTrees': {'type': 'int_uniform', 'low': 50, 'high': 300},
            'numGfr': {'type': 'int_uniform', 'low': 5, 'high': 20},
            'numSamples': {'type': 'int_uniform', 'low': 20, 'high': 200},
            'alpha': {'type': 'uniform', 'low': 0.5, 'high': 0.99},
            'beta': {'type': 'uniform', 'low': 0.5, 'high': 3.0},
            'minSamplesLeaf': {'type': 'int_uniform', 'low': 1, 'high': 20},
            'cutpointGrid': {'type': 'categorical', 'values': [50, 100, 200]},
        }


register('wlearn.stochtree.classifier@1', BARTModel._from_bundle)
register('wlearn.stochtree.regressor@1', BARTModel._from_bundle)
