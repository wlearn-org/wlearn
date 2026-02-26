"""Ensemble weight optimization via projected gradient descent on the simplex.

Refines frequency-based Caruana weights to optimized weights using gradient
descent with simplex projection (Duchi et al. 2008).
"""

import numpy as np


def project_simplex(v):
    """Project vector v onto the probability simplex {w: w >= 0, sum(w) = 1}.

    Algorithm from Duchi et al. (2008), O(n log n).

    Args:
        v: np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (n,) on the simplex
    """
    n = len(v)
    if n == 0:
        return v.copy()
    if n == 1:
        return np.array([1.0])

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho_candidates) == 0:
        # Fallback: uniform weights
        return np.full(n, 1.0 / n)
    rho = rho_candidates[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def optimize_weights(oof_predictions, y_true, init_weights,
                     task='classification', lr=0.05, n_iter=100):
    """Optimize ensemble member weights via projected gradient descent.

    For classification: minimizes negative log-loss over OOF probabilities.
    For regression: minimizes MSE over OOF predictions.
    Constraints: weights >= 0, sum(weights) = 1.

    Args:
        oof_predictions: list of np.ndarray (one per selected model)
            Classification: each (n * n_classes,) flat row-major
            Regression: each (n,)
        y_true: np.ndarray of true labels
        init_weights: np.ndarray of initial weights (from Caruana selection)
        task: 'classification' or 'regression'
        lr: learning rate for gradient descent
        n_iter: number of gradient steps

    Returns:
        np.ndarray of optimized weights (>= 0, sum = 1)
    """
    n = len(y_true)
    m = len(oof_predictions)

    if m == 0:
        return np.array([], dtype=np.float64)
    if m == 1:
        return np.array([1.0])

    w = init_weights.copy().astype(np.float64)
    # Ensure starting point is on simplex
    w = project_simplex(w)

    eps = 1e-15

    if task == 'classification':
        n_classes = len(oof_predictions[0]) // n
        # Precompute: for each model m, extract proba[i, y_true[i]]
        # and full proba matrix for gradient computation
        y_int = np.asarray(y_true, dtype=np.int32)

        for _ in range(n_iter):
            # Compute ensemble proba for true class: p_i = sum(w_m * oof_m[i*nc + y_i])
            p_true = np.zeros(n, dtype=np.float64)
            for j in range(m):
                wj = w[j]
                oof = oof_predictions[j]
                for i in range(n):
                    p_true[i] += wj * oof[i * n_classes + y_int[i]]

            # Clip for numerical stability
            np.maximum(p_true, eps, out=p_true)

            # Gradient: d(loss)/d(w_m) = -1/n * sum(oof_m[i*nc+y_i] / p_true[i])
            grad = np.zeros(m, dtype=np.float64)
            for j in range(m):
                oof = oof_predictions[j]
                g = 0.0
                for i in range(n):
                    g += oof[i * n_classes + y_int[i]] / p_true[i]
                grad[j] = -g / n

            w = project_simplex(w - lr * grad)

    else:
        # Regression: minimize MSE
        for _ in range(n_iter):
            # Compute ensemble prediction: p_i = sum(w_m * oof_m[i])
            p = np.zeros(n, dtype=np.float64)
            for j in range(m):
                wj = w[j]
                oof = oof_predictions[j]
                for i in range(n):
                    p[i] += wj * float(oof[i])

            # Residuals
            resid = np.asarray(y_true, dtype=np.float64) - p

            # Gradient: d(MSE)/d(w_m) = -2/n * sum(resid[i] * oof_m[i])
            grad = np.zeros(m, dtype=np.float64)
            for j in range(m):
                oof = oof_predictions[j]
                g = 0.0
                for i in range(n):
                    g += resid[i] * float(oof[i])
                grad[j] = -2.0 * g / n

            w = project_simplex(w - lr * grad)

    return w
