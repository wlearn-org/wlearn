"""Python wrapper for @wlearn/xlearn bundles.

Loads WLRN bundles produced by JS @wlearn/xlearn, predicts using pure numpy
(no xlearn C++ dependency needed for inference), and saves back to WLRN bundles
that JS can load.

Blob format: xlearn binary model (wasm32 size_t = 4 bytes):
  [u32]       score_func string length
  [bytes]     score_func ("linear", "fm", "ffm")
  [u32]       loss_func string length
  [bytes]     loss_func ("cross-entropy", "squared", etc.)
  [u32]       num_feat
  [u32]       num_field
  [u32]       num_K
  [u32]       aux_size
  [u32]       param_num_w
  [u32]       param_num_v (FM/FFM only, omitted for linear)
  [f32 * param_num_w]   w array (weights interleaved with gradient cache)
  [f32 * aux_size]      b array (bias + gradient cache)
  [f32 * param_num_v]   v array (latent factors + gradient cache, FM/FFM only)

Weight extraction:
  w_param[j]   = w[j * aux_size]
  b_param      = b[0]
  v_param[j,k] = v[j * aligned_K * aux_size + k]  (FM)
  v_param[j,field,k] = v[(j * num_field + field) * aligned_K * aux_size + k]  (FFM)
  aligned_K    = ceil(K / 4) * 4
"""

import math
import struct

import numpy as np

from .errors import NotFittedError, DisposedError
from ._capabilities import estimator_capabilities
from .bundle import encode_bundle, write_bundle_output
from .registry import register


K_ALIGN = 4  # SSE alignment constant


def _align_k(k):
    """Align K to SSE register boundary (multiple of 4)."""
    return ((k + K_ALIGN - 1) // K_ALIGN) * K_ALIGN


def _read_str(blob, pos):
    """Read a length-prefixed string (wasm32: u32 length prefix)."""
    if pos > len(blob) - 4:
        raise ValueError('xLearn model is truncated before a string length')
    length = struct.unpack_from('<I', blob, pos)[0]
    pos += 4
    if length < 1 or length > 64 or pos > len(blob) - length:
        raise ValueError('xLearn model has an invalid string field')
    try:
        s = blob[pos:pos + length].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('xLearn model has a non-ASCII string field') \
            from error
    pos += length
    return s, pos


def _write_str(s):
    """Write a length-prefixed string (wasm32: u32 length prefix)."""
    encoded = s.encode('ascii')
    return struct.pack('<I', len(encoded)) + encoded


def _parse_model(blob):
    """Parse xlearn binary model blob into weight arrays and metadata.

    Returns dict with keys: score_func, loss_func, num_feat, num_field,
    num_K, aux_size, w, b, v (numpy float32 arrays).
    """
    blob = bytes(blob)
    pos = 0

    score_func, pos = _read_str(blob, pos)
    loss_func, pos = _read_str(blob, pos)

    def read_u32(name):
        nonlocal pos
        if pos > len(blob) - 4:
            raise ValueError(f'xLearn model is truncated before {name}')
        value = struct.unpack_from('<I', blob, pos)[0]
        pos += 4
        return value

    num_feat = read_u32('feature count')
    num_field = read_u32('field count')
    num_K = read_u32('factor count')
    aux_size = read_u32('auxiliary parameter count')
    param_num_w = read_u32('linear weight count')

    param_num_v = 0
    if score_func != 'linear':
        param_num_v = read_u32('latent factor count')

    if (score_func not in ('linear', 'fm', 'ffm') or
            loss_func not in ('cross-entropy', 'squared') or
            num_feat < 1):
        raise ValueError('xLearn model has unsupported identity fields')
    if (aux_size < 1 or
            score_func in ('linear', 'fm') and num_field != 0 or
            score_func != 'linear' and num_K < 1 or
            score_func == 'ffm' and num_field < 1):
        raise ValueError('xLearn model has invalid dimensions')

    max_u32 = 2**32 - 1

    def checked_product(name, *values):
        product = 1
        for value in values:
            if value < 1 or product > max_u32 // value:
                raise ValueError(
                    f'xLearn model {name} exceeds uint32 limits')
            product *= value
        return product

    expected_w = checked_product('weight count', num_feat, aux_size)
    if param_num_w != expected_w:
        raise ValueError('xLearn model linear weight count is inconsistent')

    expected_v = 0
    if score_func != 'linear':
        aligned_k = _align_k(num_K)
        if score_func == 'fm':
            expected_v = checked_product(
                'latent factor count', num_feat, aligned_k, aux_size)
        else:
            expected_v = checked_product(
                'latent factor count', num_feat, num_field, aligned_k,
                aux_size)
        if param_num_v != expected_v:
            raise ValueError(
                'xLearn model latent factor count is inconsistent')

    expected_bytes = pos + 4 * (expected_w + aux_size + expected_v)
    if expected_bytes != len(blob):
        raise ValueError(
            'xLearn model byte length is inconsistent with its parameter counts')

    w = np.frombuffer(blob, dtype='<f4', count=param_num_w, offset=pos).copy()
    pos += param_num_w * 4

    b = np.frombuffer(blob, dtype='<f4', count=aux_size, offset=pos).copy()
    pos += aux_size * 4

    v = None
    if param_num_v > 0:
        v = np.frombuffer(blob, dtype='<f4', count=param_num_v, offset=pos).copy()
        pos += param_num_v * 4

    return {
        'score_func': score_func,
        'loss_func': loss_func,
        'num_feat': num_feat,
        'num_field': num_field,
        'num_K': num_K,
        'aux_size': aux_size,
        'w': w,
        'b': b,
        'v': v,
    }


def _serialize_model(model):
    """Serialize model dict back to xlearn binary format."""
    parts = []
    parts.append(_write_str(model['score_func']))
    parts.append(_write_str(model['loss_func']))
    parts.append(struct.pack('<I', model['num_feat']))
    parts.append(struct.pack('<I', model['num_field']))
    parts.append(struct.pack('<I', model['num_K']))
    parts.append(struct.pack('<I', model['aux_size']))
    parts.append(struct.pack('<I', len(model['w'])))
    if model['score_func'] != 'linear':
        v = model['v']
        parts.append(struct.pack('<I', len(v) if v is not None else 0))
    parts.append(model['w'].astype('<f4').tobytes())
    parts.append(model['b'].astype('<f4').tobytes())
    if model['v'] is not None:
        parts.append(model['v'].astype('<f4').tobytes())
    return b''.join(parts)


def _extract_weights(model):
    """Extract actual parameter weights (strip gradient cache).

    Returns: w_params (num_feat,), bias (scalar), v_params (num_feat, K) or None
    """
    aux = model['aux_size']
    num_feat = model['num_feat']

    # w: params at stride positions
    w_params = np.array([model['w'][j * aux] for j in range(num_feat)],
                        dtype=np.float32)
    bias = float(model['b'][0])

    v_params = None
    if model['v'] is not None:
        K = model['num_K']
        aligned_K = _align_k(K)
        score_func = model['score_func']

        if score_func == 'fm':
            v_params = np.zeros((num_feat, K), dtype=np.float32)
            for j in range(num_feat):
                base = j * aligned_K * aux
                for k in range(K):
                    v_params[j, k] = model['v'][base + k]
        elif score_func == 'ffm':
            num_field = model['num_field']
            v_params = np.zeros((num_feat, num_field, K), dtype=np.float32)
            for j in range(num_feat):
                for f in range(num_field):
                    base = (j * num_field + f) * aligned_K * aux
                    for k in range(K):
                        v_params[j, f, k] = model['v'][base + k]

    return w_params, bias, v_params


def _compute_norm(X):
    """Compute instance-wise L2 norm factor: 1.0 / sum(x^2).

    Returns array of shape (n,). Zero-norm rows get 1.0.
    """
    sq_sum = np.sum(X * X, axis=1)
    return np.where(sq_sum > 0, 1.0 / sq_sum, 1.0)


def _validate_matrix(X, expected_features):
    try:
        matrix = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError('X must be a rectangular numeric 2-D matrix') \
            from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            'X must be a non-empty rectangular numeric 2-D matrix')
    if matrix.shape[1] != expected_features:
        raise ValueError(
            f'X has {matrix.shape[1]} features; model expects '
            f'{expected_features}')
    if not np.isfinite(matrix).all():
        raise ValueError('X values must be finite')
    return matrix


def _predict_linear(X, w, bias, normalize):
    """Linear prediction: score = w^T x + b.

    xlearn's linear scoring ignores the norm factor entirely.
    """
    scores = np.dot(X, w).astype(np.float64)
    scores += bias
    return scores


def _predict_fm(X, w, bias, v, K, normalize):
    """FM prediction with instance-wise normalization.

    xlearn applies norm as:
      linear_term = sum_j(w_j * x_j * sqrt(norm))
      interaction = 0.5 * sum_f[ (sum_j v_jf*x_j*norm)^2 - sum_j (v_jf*x_j*norm)^2 ]
      score = linear_term + bias + interaction
    """
    n = X.shape[0]

    if normalize:
        norm = _compute_norm(X)  # (n,)
        sqrt_norm = np.sqrt(norm)  # (n,)

        # Linear: x * sqrt(norm)
        Xw = X * sqrt_norm[:, None]
        linear = np.dot(Xw, w).astype(np.float64)

        # Interaction: x * norm
        Xn = X * norm[:, None]
        Xnv = np.dot(Xn, v)  # (n, K)
        Xn2v2 = np.dot(Xn * Xn, v * v)  # (n, K)
    else:
        linear = np.dot(X, w).astype(np.float64)
        Xnv = np.dot(X, v)
        Xn2v2 = np.dot(X * X, v * v)

    interaction = np.sum(Xnv * Xnv - Xn2v2, axis=1)
    scores = linear + bias + 0.5 * interaction
    return scores


def _predict_ffm(X, w, bias, v, K, num_field, field_map, normalize):
    """FFM prediction with field-aware interactions.

    Norm applied same as FM: sqrt(norm) on linear, norm on interaction.
    """
    n, d = X.shape

    if normalize:
        norm = _compute_norm(X)
        sqrt_norm = np.sqrt(norm)
        Xw = X * sqrt_norm[:, None]
        linear = np.dot(Xw, w).astype(np.float64)
    else:
        norm = np.ones(n)
        linear = np.dot(X, w).astype(np.float64)

    scores = linear + bias

    # Pairwise field-aware interactions
    for i in range(n):
        ni = float(norm[i]) if normalize else 1.0
        interaction = 0.0
        for j1 in range(d):
            if X[i, j1] == 0:
                continue
            f1 = field_map[j1] if field_map is not None else 0
            for j2 in range(j1 + 1, d):
                if X[i, j2] == 0:
                    continue
                f2 = field_map[j2] if field_map is not None else 0
                dot = 0.0
                for k in range(K):
                    dot += float(v[j1, f2, k]) * float(v[j2, f1, k])
                interaction += dot * float(X[i, j1]) * float(X[i, j2]) * ni * ni
        scores[i] += interaction

    return scores


class XLearnModel:
    def __init__(self, raw_model, params, metadata, raw_blob=None,
                 type_id=None, manifest_seed=None):
        self._raw_model = raw_model  # parsed model dict
        self._params = dict(params)
        self._disposed = False
        self._fitted = True
        self._raw_blob = raw_blob
        self._type_id = type_id
        self._manifest_seed = manifest_seed

        self._algo = metadata.get('algo', raw_model['score_func'])
        self._task = metadata.get('task', 'binary')
        self._num_features = raw_model['num_feat']
        self._nr_class = metadata.get('nClasses', 0)
        classes = metadata.get('classes')
        self._classes = np.array(classes, dtype=np.int32) if classes else None
        self._field_map = None

        # Extract usable weights
        self._w, self._bias, self._v = _extract_weights(raw_model)

    @classmethod
    def create(cls, params=None):
        """Create an unfitted XLearnModel."""
        obj = cls.__new__(cls)
        obj._raw_model = None
        obj._params = dict(params) if params else {}
        obj._disposed = False
        obj._fitted = False
        obj._raw_blob = None
        obj._type_id = None
        obj._manifest_seed = None
        obj._algo = obj._params.get('algo', 'fm')
        obj._task = 'binary'
        obj._num_features = 0
        obj._nr_class = 0
        obj._classes = None
        obj._field_map = None
        obj._w = None
        obj._bias = 0.0
        obj._v = None
        return obj

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        type_id = manifest.get('typeId')
        spec = _bundle_spec(type_id)
        metadata = manifest.get('metadata', {})
        if metadata.get('algo') != spec['algo'] or \
                metadata.get('task') != spec['task']:
            raise ValueError(
                f'{type_id} metadata does not match '
                f'{spec["algo"]}/{spec["task"]}')
        n_features = metadata.get('nFeatures')
        if (isinstance(n_features, bool) or not isinstance(n_features, int) or
                n_features < 1):
            raise ValueError(f'{type_id} has invalid nFeatures metadata')
        if spec['task'] == 'reg' and (
                metadata.get('nClasses') != 0 or
                metadata.get('classes') is not None):
            raise ValueError(f'{type_id} has classifier metadata')

        entry = next((e for e in toc if e['id'] == 'model'), None)
        field_entries = [e for e in toc if e['id'] == 'field_map']
        ffm_with_map = (
            spec['algo'] == 'ffm' and len(field_entries) == 1 and
            len(toc) == 2)
        if (entry is None or len([e for e in toc if e['id'] == 'model']) != 1 or
                not (len(toc) == 1 or ffm_with_map) or
                any(e.get('mediaType') != 'application/octet-stream'
                    for e in toc) or
                any(e['id'] not in ('model', 'field_map') for e in toc) or
                (spec['algo'] != 'ffm' and field_entries)):
            raise ValueError(f'{type_id} has an invalid artifact set')

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        raw_model = _parse_model(blob)

        if raw_model['score_func'] != spec['algo']:
            raise ValueError(
                f'{type_id} model blob algorithm does not match its typeId')
        expected_loss = (
            'cross-entropy' if spec['task'] == 'binary' else 'squared')
        if raw_model['loss_func'] != expected_loss:
            raise ValueError(
                f'{type_id} model loss does not match its task')
        if raw_model['num_feat'] != n_features:
            raise ValueError(
                f'{type_id} model feature count does not match metadata')

        params = dict(manifest.get('params', {}))
        if 'seed' in manifest:
            seed = manifest['seed']
            if (not isinstance(seed, int) or isinstance(seed, bool) or
                    seed < 1 or seed > 2147483647 or
                    params.get('seed') != seed):
                raise ValueError(f'{type_id} has invalid training seed metadata')
        if spec['version'] == 2:
            _validate_v2_classifier_metadata(type_id, metadata)
            if 'seed' not in manifest:
                raise ValueError(f'{type_id} is missing its training seed')
        elif spec['task'] == 'binary':
            _validate_legacy_classifier_metadata(type_id, metadata)

        obj = XLearnModel(
            raw_model, params, metadata, raw_blob=blob, type_id=type_id,
            manifest_seed=manifest.get('seed'))

        # Load field_map if present (FFM)
        field_entry = field_entries[0] if field_entries else None
        if field_entry is not None:
            if spec['algo'] != 'ffm':
                raise ValueError('Non-FFM bundle must not contain field_map')
            raw = blobs[field_entry['offset']:
                        field_entry['offset'] + field_entry['length']]
            if len(raw) != obj._num_features * 4:
                raise ValueError('field_map length must match nFeatures')
            obj._field_map = np.frombuffer(
                bytes(raw), dtype='<i4').copy()
            if (np.any(obj._field_map < 0) or
                    np.any(obj._field_map >= raw_model['num_field'])):
                raise ValueError('field_map contains an invalid field ID')
        elif spec['algo'] == 'ffm':
            if spec['version'] == 2:
                raise ValueError('FFM @2 bundle missing field_map')
            obj._field_map = np.zeros(obj._num_features, dtype=np.int32)

        return obj

    def decision_function(self, X):
        self._ensure_fitted()
        X = _validate_matrix(X, self._num_features)

        normalize = self._params.get('normalize', True)
        score_func = self._raw_model['score_func']

        if score_func == 'linear':
            return _predict_linear(X, self._w, self._bias, normalize)
        elif score_func == 'fm':
            K = self._raw_model['num_K']
            return _predict_fm(X, self._w, self._bias, self._v, K, normalize)
        elif score_func == 'ffm':
            K = self._raw_model['num_K']
            num_field = self._raw_model['num_field']
            field_map = self._field_map
            return _predict_ffm(
                X, self._w, self._bias, self._v, K,
                num_field, field_map, normalize)
        else:
            raise ValueError(f'Unknown score function: {score_func}')

    def predict(self, X):
        margins = self.decision_function(X)
        if self._task != 'binary':
            return margins
        return np.where(
            margins > 0, self._classes[1], self._classes[0]
        ).astype(np.int32)

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'binary':
            raise ValueError('predict_proba only for classification')

        margins = self.decision_function(X)
        n = len(margins)
        proba = np.empty(n * 2, dtype=np.float64)
        for i in range(n):
            margin = float(margins[i])
            if margin >= 0:
                exp_value = math.exp(-margin)
                p1 = 1.0 / (1.0 + exp_value)
            else:
                exp_value = math.exp(margin)
                p1 = exp_value / (1.0 + exp_value)
            proba[i * 2] = 1.0 - p1
            proba[i * 2 + 1] = p1
        return proba

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y)
        if y.size != preds.size:
            raise ValueError(
                f'y length ({y.size}) does not match predictions '
                f'({preds.size})')
        if self._task == 'binary':
            return float(np.mean(preds == y))
        # R-squared
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self, path=None):
        self._ensure_fitted()
        if self._raw_blob is not None:
            model_blob = self._raw_blob
        else:
            model_blob = _serialize_model(self._raw_model)

        artifacts = [{'id': 'model', 'data': model_blob}]

        # FFM field map
        if self._algo == 'ffm' and self._field_map is not None:
            artifacts.append({
                'id': 'field_map',
                'data': self._field_map.astype('<i4').tobytes(),
            })

        type_id = self._type_id or _type_id(self._algo, self._task)

        metadata = {
            'algo': self._algo,
            'task': self._task,
            'nFeatures': int(self._num_features),
            'nClasses': int(self._nr_class),
            'classes': (self._classes.tolist()
                        if self._classes is not None else None),
        }
        if type_id.endswith('@2'):
            metadata['labelEncoding'] = 'sorted-int32-sign-v1'

        manifest = {
            'typeId': type_id,
            'params': self.get_params(),
            'metadata': metadata,
        }
        if self._manifest_seed is not None:
            manifest['seed'] = self._manifest_seed
        bundle = encode_bundle(manifest, artifacts)
        return write_bundle_output(bundle, path)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._fitted = False
        self._raw_model = None
        self._w = None
        self._v = None

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
        classifier = self._task == 'binary'
        return estimator_capabilities(
            classifier=classifier,
            regressor=not classifier,
            predict_proba=classifier,
            decisionFunction=True,
        )

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('XLearnModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('XLearnModel is not fitted.')

    @classmethod
    def default_search_space(cls):
        return {
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1.0},
            'lambda': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-1},
            'k': {'type': 'int_uniform', 'low': 2, 'high': 16},
            'epoch': {'type': 'int_uniform', 'low': 5, 'high': 50},
            'opt': {'type': 'categorical', 'values': ['adagrad', 'ftrl', 'sgd']},
        }


def _type_id(algo, task):
    """Map (algo, task) to wlearn typeId."""
    algo_map = {'linear': 'lr', 'fm': 'fm', 'ffm': 'ffm'}
    algo_short = algo_map.get(algo, algo)
    task_str = 'classifier' if task == 'binary' else 'regressor'
    version = 2 if task == 'binary' else 1
    return f'wlearn.xlearn.{algo_short}.{task_str}@{version}'


def _bundle_spec(type_id):
    specs = {}
    for algo_short, algo in (('lr', 'linear'), ('fm', 'fm'), ('ffm', 'ffm')):
        specs[f'wlearn.xlearn.{algo_short}.classifier@1'] = {
            'algo': algo, 'task': 'binary', 'version': 1}
        specs[f'wlearn.xlearn.{algo_short}.classifier@2'] = {
            'algo': algo, 'task': 'binary', 'version': 2}
        specs[f'wlearn.xlearn.{algo_short}.regressor@1'] = {
            'algo': algo, 'task': 'reg', 'version': 1}
    try:
        return specs[type_id]
    except KeyError as error:
        raise ValueError(f'Unsupported xLearn bundle typeId: {type_id}') \
            from error


def _validated_classes(type_id, metadata):
    classes = metadata.get('classes')
    if (metadata.get('nClasses') != 2 or not isinstance(classes, list) or
            len(classes) != 2 or
            any(isinstance(value, bool) or not isinstance(value, int) or
                value < -2147483648 or value > 2147483647
                for value in classes) or
            classes[0] >= classes[1]):
        raise ValueError(f'{type_id} has invalid binary class metadata')
    return classes


def _validate_v2_classifier_metadata(type_id, metadata):
    _validated_classes(type_id, metadata)
    if metadata.get('labelEncoding') != 'sorted-int32-sign-v1':
        raise ValueError(f'{type_id} has invalid label encoding')


def _validate_legacy_classifier_metadata(type_id, metadata):
    classes = _validated_classes(type_id, metadata)
    if not classes[0] <= 0 < classes[1]:
        raise ValueError(
            f'{type_id} used an ambiguous same-sign label mapping; '
            'retrain the model')


# Register the three corrected classifier IDs plus legacy readers and regressors.
for _algo in ('lr', 'fm', 'ffm'):
    register(f'wlearn.xlearn.{_algo}.classifier@1', XLearnModel._from_bundle)
    register(f'wlearn.xlearn.{_algo}.classifier@2', XLearnModel._from_bundle)
    register(f'wlearn.xlearn.{_algo}.regressor@1', XLearnModel._from_bundle)
