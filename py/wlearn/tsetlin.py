"""Python wrapper for @wlearn/tsetlin bundles.

Loads WLRN bundles produced by JS @wlearn/tsetlin, predicts using pure numpy
clause evaluation, and saves back to WLRN bundles that JS can load.

fit() uses tmu (Tsetlin Machine Unified) for training -- lazy import.
predict() uses pure numpy (no tmu dependency).

Blob format (TM01):
  [0..3]    "TM01" magic
  [4..7]    version=1 (u32 LE)
  [8..11]   n_clauses
  [12..15]  n_features
  [16..19]  n_classes
  [20..23]  n_binary
  [24..27]  state_bits
  [28..31]  threshold
  [32..39]  s (f64 LE)
  [40]      task (0=cls, 1=reg)
  [41]      boost_tpf
  [42..43]  n_thresholds_per_feature (u16 LE)
  [44..47]  reserved
  [48..55]  y_min (f64 LE)
  [56..63]  y_max (f64 LE)
  [64..)    threshold_counts[n_features] (i32 LE)
  [..)      thresholds (f64 LE, packed per feature)
  [..)      ta_state (u32 LE array)
  [..)      class_labels[n_classes] (i32 LE, cls only)
"""

import struct

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

TM01_MAGIC = b'TM01'
TM01_HEADER = 64
TM01_HEADER_FMT = '<4sIIIIIII8sBBH4x8s8s'  # 64 bytes


def _binarize(X, thresholds, threshold_counts, n_thresholds_per_feature):
    """Binarize feature matrix using stored thresholds.

    Returns a 2D uint32 array of shape (n_samples, la_chunks) with bit-packed literals.
    """
    n_samples, n_features = X.shape
    n_binary = int(np.sum(threshold_counts))
    n_literals = 2 * n_binary
    la_chunks = (n_literals - 1) // 32 + 1

    Xi_all = np.zeros((n_samples, la_chunks), dtype=np.uint32)

    # Set positive literals
    bit_pos = 0
    for f in range(n_features):
        cnt = threshold_counts[f]
        for t in range(cnt):
            thresh_val = thresholds[f * n_thresholds_per_feature + t]
            mask = X[:, f] > thresh_val
            chunk = bit_pos // 32
            bit = bit_pos % 32
            Xi_all[mask, chunk] |= np.uint32(1 << bit)
            bit_pos += 1

    # Set negated literals
    for f in range(n_features):
        cnt = threshold_counts[f]
        for t in range(cnt):
            thresh_val = thresholds[f * n_thresholds_per_feature + t]
            mask = X[:, f] <= thresh_val
            chunk = bit_pos // 32
            bit = bit_pos % 32
            Xi_all[mask, chunk] |= np.uint32(1 << bit)
            bit_pos += 1

    return Xi_all, n_binary, n_literals, la_chunks


def _evaluate_clauses(ta_state, Xi, n_clauses, n_literals, state_bits, la_chunks):
    """Evaluate all clauses for a single sample. Returns clause_output array."""
    n_ta_chunks = la_chunks
    clause_output = np.zeros(n_clauses, dtype=np.int32)

    # Filter mask for last chunk
    if n_literals % 32 != 0:
        filt = np.uint32((1 << (n_literals % 32)) - 1)
    else:
        filt = np.uint32(0xFFFFFFFF)

    for j in range(n_clauses):
        clause_pos = j * n_ta_chunks * state_bits
        output = True
        all_exclude = True

        for k in range(n_ta_chunks):
            pos = clause_pos + k * state_bits + state_bits - 1
            action_bits = ta_state[pos]

            if k < n_ta_chunks - 1:
                # Check: included literals must match Xi
                if (action_bits & Xi[k]) != action_bits:
                    output = False
                    break
                if action_bits != 0:
                    all_exclude = False
            else:
                # Last chunk: apply filter
                masked_action = action_bits & filt
                if (masked_action & Xi[k]) != masked_action:
                    output = False
                    break
                if masked_action != 0:
                    all_exclude = False

        if output and not all_exclude:
            clause_output[j] = 1

    return clause_output


def _predict_votes(ta_state, Xi_all, n_classes, n_clauses, n_literals,
                   state_bits, la_chunks):
    """Compute vote sums for each sample and each class."""
    n_samples = Xi_all.shape[0]
    clause_state_size = n_clauses * la_chunks * state_bits
    half_clauses = n_clauses // 2
    votes = np.zeros((n_samples, n_classes), dtype=np.int32)

    for i in range(n_samples):
        Xi = Xi_all[i]
        for c in range(n_classes):
            ta_c = ta_state[c * clause_state_size:(c + 1) * clause_state_size]
            co = _evaluate_clauses(ta_c, Xi, n_clauses, n_literals,
                                   state_bits, la_chunks)
            vote_sum = int(np.sum(co[:half_clauses])) - int(np.sum(co[half_clauses:]))
            votes[i, c] = vote_sum

    return votes


_tmu_patched = False


def _patch_tmu_numpy_compat():
    """Patch tmu's use of np.uint32(~0) for numpy >= 2.0 compatibility.

    tmu 0.7.1 uses ``np.uint32(~0)`` and ``array | ~0`` which fail with
    numpy >= 2.0 because negative Python ints can no longer be cast to
    unsigned numpy types.  We replace those patterns with the equivalent
    ``np.uint32(0xFFFFFFFF)`` / ``np.full(..., 0xFFFFFFFF, np.uint32)``.
    """
    global _tmu_patched
    if _tmu_patched:
        return
    _tmu_patched = True

    try:
        np.uint32(~0)
        return  # numpy version is fine, no patch needed
    except (OverflowError, TypeError):
        pass

    import importlib
    import re

    pkg_names = ['tmu.clause_bank', 'tmu.tsetlin_machine']
    for name in pkg_names:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        src_path = mod.__file__
        if src_path is None:
            continue
        with open(src_path, 'r') as f:
            src = f.read()

        patched = src.replace('np.uint32(~0)', 'np.uint32(0xFFFFFFFF)')
        # Fix: (np.zeros(..., dtype=np.uint32) | ~0).astype(np.uint32)
        # ->   np.full(..., 0xFFFFFFFF, dtype=np.uint32)
        patched = re.sub(
            r'\(np\.zeros\(([^)]+),\s*dtype=np\.uint32\)\s*\|\s*~0\)\.astype\(np\.uint32\)',
            r'np.full(\1, 0xFFFFFFFF, dtype=np.uint32)',
            patched,
        )
        if patched != src:
            code = compile(patched, src_path, 'exec')
            mod_ns = mod.__dict__
            exec(code, mod_ns)


class TsetlinModel:
    def __init__(self, ta_state, thresholds, threshold_counts, params,
                 class_labels=None, metadata=None):
        self._ta_state = ta_state
        self._thresholds = thresholds
        self._threshold_counts = threshold_counts
        self._params = dict(params)
        self._class_labels = class_labels
        self._disposed = False
        self._fitted = True

        md = metadata or {}
        self._n_classes = md.get('n_classes', 0)
        self._n_features = md.get('n_features', 0)
        self._n_binary = md.get('n_binary', 0)
        self._n_clauses = md.get('n_clauses', 0)
        self._n_literals = md.get('n_literals', 0)
        self._la_chunks = md.get('la_chunks', 0)
        self._state_bits = md.get('state_bits', 8)
        self._threshold_val = md.get('threshold', 50)
        self._s = md.get('s', 3.0)
        self._task = md.get('task', 0)
        self._boost_tpf = md.get('boost_tpf', 0)
        self._n_thresholds_per_feature = md.get('n_thresholds_per_feature', 10)
        self._y_min = md.get('y_min', 0.0)
        self._y_max = md.get('y_max', 1.0)

        if class_labels is not None:
            self._classes = np.array(class_labels, dtype=np.int32)
        else:
            self._classes = None

    @classmethod
    def create(cls, params=None):
        """Create an unfitted TsetlinModel."""
        obj = cls.__new__(cls)
        obj._ta_state = None
        obj._thresholds = None
        obj._threshold_counts = None
        obj._params = dict(params) if params else {}
        obj._class_labels = None
        obj._classes = None
        obj._disposed = False
        obj._fitted = False
        obj._n_classes = 0
        obj._n_features = 0
        obj._n_binary = 0
        obj._n_clauses = 0
        obj._n_literals = 0
        obj._la_chunks = 0
        obj._state_bits = 8
        obj._threshold_val = 50
        obj._s = 3.0
        obj._task = 0
        obj._boost_tpf = 0
        obj._n_thresholds_per_feature = 10
        obj._y_min = 0.0
        obj._y_max = 1.0
        return obj

    def fit(self, X, y):
        """Train a Tsetlin Machine using the tmu package.

        Requires the ``tmu`` package (``pip install tmu>=0.7``).
        Predict/save/load only need numpy.
        """
        if self._disposed:
            raise DisposedError('TsetlinModel has been disposed.')

        try:
            _patch_tmu_numpy_compat()
            from tmu.tsetlin_machine import TMClassifier, TMRegressor
        except ImportError:
            raise ImportError(
                'tmu package required for fit(). '
                'Install with: pip install tmu>=0.7'
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_clauses = self._params.get('nClauses', 100)
        threshold = self._params.get('threshold', 50)
        s = self._params.get('s', 3.0)
        state_bits = self._params.get('stateBits', 8)
        boost = self._params.get('boostTruePositiveFeedback', False)
        n_thresh = self._params.get('nThresholdsPerFeature', 10)
        n_epochs = self._params.get('nEpochs', 100)
        seed = self._params.get('seed', 42)
        task_str = self._params.get('task', 'classification')

        # Binarize: compute thresholds from training data
        n_features = X.shape[1]
        thresholds_list = []
        threshold_counts = np.zeros(n_features, dtype=np.int32)

        for f in range(n_features):
            col = np.sort(np.unique(X[:, f]))
            if len(col) <= 1:
                thresholds_list.append(np.array([], dtype=np.float64))
                continue
            n_t = min(n_thresh, len(col) - 1)
            quantiles = np.linspace(0, 1, n_t + 2)[1:-1]
            t_vals = np.quantile(X[:, f], quantiles)
            t_vals = np.unique(t_vals)
            thresholds_list.append(t_vals)
            threshold_counts[f] = len(t_vals)

        # Pack thresholds into flat array
        all_thresholds = np.zeros(n_features * n_thresh, dtype=np.float64)
        for f in range(n_features):
            for t in range(threshold_counts[f]):
                all_thresholds[f * n_thresh + t] = thresholds_list[f][t]

        n_binary = int(np.sum(threshold_counts))
        n_literals = 2 * n_binary
        la_chunks = (n_literals - 1) // 32 + 1

        # Binarize training data to unpacked binary features for tmu
        # tmu expects shape (n_samples, n_binary) with 0/1 uint32 values
        Xi_unpacked = np.zeros((X.shape[0], n_binary), dtype=np.uint32)
        bit_pos = 0
        for f in range(n_features):
            cnt = threshold_counts[f]
            for t in range(cnt):
                thresh_val = all_thresholds[f * n_thresh + t]
                Xi_unpacked[:, bit_pos] = (X[:, f] > thresh_val).astype(np.uint32)
                bit_pos += 1

        # Set numpy random seed for reproducibility (tmu uses numpy PRNG)
        np.random.seed(seed)

        # Use tmu for training (tmu does its own bit-packing internally)
        Xi_uint = Xi_unpacked

        if task_str == 'regression':
            tm = TMRegressor(
                number_of_clauses=n_clauses,
                T=threshold,
                s=s,
                platform='CPU',
                boost_true_positive_feedback=1 if boost else 0,
                number_of_state_bits_ta=state_bits,
            )
            y_float = y.astype(np.float32)
            for ep in range(n_epochs):
                tm.fit(Xi_uint, y_float)

            self._task = 1
            self._y_min = float(tm.min_y)
            self._y_max = float(tm.max_y)
            self._n_classes = 1
            self._class_labels = None
            self._classes = None

            # TMRegressor: single clause_bank
            banks = [tm.clause_bank]

        else:
            tm = TMClassifier(
                number_of_clauses=n_clauses,
                T=threshold,
                s=s,
                platform='CPU',
                boost_true_positive_feedback=1 if boost else 0,
                number_of_state_bits_ta=state_bits,
            )
            y_int = y.astype(np.uint32)
            for ep in range(n_epochs):
                tm.fit(Xi_uint, y_int)

            self._task = 0
            unique_labels = np.sort(np.unique(y.astype(np.int32)))
            self._class_labels = unique_labels.tolist()
            self._classes = unique_labels
            self._n_classes = len(unique_labels)
            self._y_min = 0.0
            self._y_max = 0.0

            # TMClassifier: one clause_bank per class
            banks = tm.clause_banks

        # Extract TA state from tmu internals
        clause_state_size = n_clauses * la_chunks * state_bits
        total_ta = self._n_classes * clause_state_size
        ta_state = np.zeros(total_ta, dtype=np.uint32)

        for c_idx, bank in enumerate(banks):
            bank_ta = np.asarray(bank.clause_bank, dtype=np.uint32).ravel()
            start = c_idx * clause_state_size
            copy_len = min(len(bank_ta), clause_state_size)
            ta_state[start:start + copy_len] = bank_ta[:copy_len]

        self._ta_state = ta_state
        self._thresholds = all_thresholds
        self._threshold_counts = threshold_counts
        self._n_features = n_features
        self._n_binary = n_binary
        self._n_clauses = n_clauses
        self._n_literals = n_literals
        self._la_chunks = la_chunks
        self._state_bits = state_bits
        self._threshold_val = threshold
        self._s = s
        self._boost_tpf = 1 if boost else 0
        self._n_thresholds_per_feature = n_thresh
        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        if len(blob) < TM01_HEADER:
            raise ValueError('Blob too short for TM01 header')

        # Parse header
        parts = struct.unpack_from(TM01_HEADER_FMT, blob, 0)
        magic = parts[0]
        version = parts[1]
        n_clauses = parts[2]
        n_features = parts[3]
        n_classes = parts[4]
        n_binary = parts[5]
        state_bits = parts[6]
        threshold = parts[7]
        s = struct.unpack('<d', parts[8])[0]
        task = parts[9]
        boost_tpf = parts[10]
        n_thresholds_per_feature = parts[11]
        y_min = struct.unpack('<d', parts[12])[0]
        y_max = struct.unpack('<d', parts[13])[0]

        if magic != TM01_MAGIC:
            raise ValueError(f'Invalid blob magic: {magic!r}')
        if version != 1:
            raise ValueError(f'Unsupported TM01 version: {version}')

        n_literals = 2 * n_binary
        la_chunks = (n_literals - 1) // 32 + 1

        pos = TM01_HEADER

        # threshold_counts
        threshold_counts = np.frombuffer(blob, dtype='<i4', count=n_features,
                                         offset=pos).copy()
        pos += n_features * 4

        # thresholds (packed)
        total_thresh = int(np.sum(threshold_counts))
        thresholds = np.zeros(n_features * n_thresholds_per_feature,
                              dtype=np.float64)
        for f in range(n_features):
            cnt = threshold_counts[f]
            if cnt > 0:
                vals = np.frombuffer(blob, dtype='<f8', count=cnt, offset=pos)
                thresholds[f * n_thresholds_per_feature:
                           f * n_thresholds_per_feature + cnt] = vals
                pos += cnt * 8

        # ta_state
        clause_state_size = n_classes * n_clauses * la_chunks * state_bits
        ta_state = np.frombuffer(blob, dtype='<u4', count=clause_state_size,
                                 offset=pos).copy()
        pos += clause_state_size * 4

        # class_labels
        class_labels = None
        if task == 0:
            class_labels = np.frombuffer(blob, dtype='<i4', count=n_classes,
                                         offset=pos).copy().tolist()

        params = manifest.get('params', {})
        metadata = {
            'n_classes': n_classes,
            'n_features': n_features,
            'n_binary': n_binary,
            'n_clauses': n_clauses,
            'n_literals': n_literals,
            'la_chunks': la_chunks,
            'state_bits': state_bits,
            'threshold': threshold,
            's': s,
            'task': task,
            'boost_tpf': boost_tpf,
            'n_thresholds_per_feature': n_thresholds_per_feature,
            'y_min': y_min,
            'y_max': y_max,
        }

        return TsetlinModel(ta_state, thresholds, threshold_counts, params,
                            class_labels=class_labels, metadata=metadata)

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xi_all, _, _, _ = _binarize(
            X, self._thresholds, self._threshold_counts,
            self._n_thresholds_per_feature
        )

        votes = _predict_votes(
            self._ta_state, Xi_all, self._n_classes, self._n_clauses,
            self._n_literals, self._state_bits, self._la_chunks
        )

        if self._task == 0:
            # Classification: argmax
            best_idx = np.argmax(votes, axis=1)
            labels = np.array(self._class_labels, dtype=np.int32)
            return labels[best_idx].astype(np.float64)
        else:
            # Regression: scale vote sum back
            raw = np.clip(votes[:, 0], 0, self._threshold_val)
            return self._y_min + raw / self._threshold_val * (self._y_max - self._y_min)

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 0:
            raise ValueError('predict_proba only for classification')

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xi_all, _, _, _ = _binarize(
            X, self._thresholds, self._threshold_counts,
            self._n_thresholds_per_feature
        )

        votes = _predict_votes(
            self._ta_state, Xi_all, self._n_classes, self._n_clauses,
            self._n_literals, self._state_bits, self._la_chunks
        )

        # Softmax
        votes_f = votes.astype(np.float64)
        max_v = np.max(votes_f, axis=1, keepdims=True)
        exp_v = np.exp(votes_f - max_v)
        proba = exp_v / np.sum(exp_v, axis=1, keepdims=True)
        return proba.ravel()

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if self._task != 0:
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()

        # Build TM01 blob
        s_bytes = struct.pack('<d', self._s)
        y_min_bytes = struct.pack('<d', self._y_min)
        y_max_bytes = struct.pack('<d', self._y_max)

        header = struct.pack(
            TM01_HEADER_FMT,
            TM01_MAGIC,
            1,  # version
            self._n_clauses,
            self._n_features,
            self._n_classes,
            self._n_binary,
            self._state_bits,
            self._threshold_val,
            s_bytes,
            self._task,
            self._boost_tpf,
            self._n_thresholds_per_feature,
            y_min_bytes,
            y_max_bytes,
        )

        parts = [header]

        # threshold_counts
        parts.append(self._threshold_counts.astype('<i4').tobytes())

        # thresholds (packed)
        for f in range(self._n_features):
            cnt = self._threshold_counts[f]
            if cnt > 0:
                start = f * self._n_thresholds_per_feature
                vals = self._thresholds[start:start + cnt]
                parts.append(np.asarray(vals, dtype='<f8').tobytes())

        # ta_state
        parts.append(np.asarray(self._ta_state, dtype='<u4').tobytes())

        # class_labels
        if self._task == 0 and self._class_labels is not None:
            parts.append(np.array(self._class_labels, dtype='<i4').tobytes())

        model_blob = b''.join(parts)

        task_str = self._params.get('task', 'classification')
        type_id = ('wlearn.tsetlin.regressor@1'
                   if task_str == 'regression'
                   else 'wlearn.tsetlin.classifier@1')

        metadata = {}
        if self._task == 0:
            metadata['nClasses'] = self._n_classes
            metadata['classes'] = (self._classes.tolist()
                                   if self._classes is not None else [])

        return encode_bundle(
            {'typeId': type_id, 'params': self.get_params(),
             'metadata': metadata},
            [{'id': 'model', 'data': model_blob}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._ta_state = None
        self._thresholds = None
        self._threshold_counts = None
        self._fitted = False

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
            raise DisposedError('TsetlinModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('TsetlinModel is not fitted.')


register('wlearn.tsetlin.classifier@1', TsetlinModel._from_bundle)
register('wlearn.tsetlin.regressor@1', TsetlinModel._from_bundle)
