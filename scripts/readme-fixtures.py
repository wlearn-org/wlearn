# Input data only: no wlearn imports or replacement model APIs.
import json as _json
try:
    import numpy as _np
except ImportError:
    _np = None

def evaluate(params):
    return -sum(v * v for v in params.values() if isinstance(v, (int, float)))

if _np is not None:
    def _rows(start, n):
        return _np.asarray([[1+j/200 if j % 2 else -1-j/200,
                             (j % 7)/7, (j % 11)/11, (j % 5)/5]
                            for j in range(start, start+n)])
    X = X_train = XTrain = _rows(0, 80)
    y = y_train = yTrain = (X[:, 0] > 0).astype(_np.int32)
    X_test = Xtest = X_new = XTest = _rows(160, 8)
    y_test = test_y = (X_test[:, 0] > 0).astype(_np.int32)
    X_imbalanced = X_large = X
    y_imbalanced = y_large = y
    calibration_probabilities = _np.asarray([[.1, .9], [.8, .2]] * 40)
    calibration_labels = _np.asarray([[0, 1], [1, 0]] * 40)
    test_probabilities = _np.asarray([[.3, .7], [.8, .2]])
