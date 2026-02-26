"""AutoML benchmark: wlearn vs TPOT vs AutoGluon.

Datasets: Friedman 1-3 (regression), moons + hastie (classification).
Sizes: 500, 2000, 10000 samples each.
"""

import time
import warnings

import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_moons
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# --- wlearn imports ---

from wlearn.xgboost import XGBModel
from wlearn.liblinear import LinearModel
from wlearn.libsvm import SVMModel
from wlearn.nanoflann import KNNModel
from wlearn.ebm import EBMModel
from wlearn.automl import auto_fit

# --- Optional competitors ---

try:
    from tpot import TPOTRegressor, TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    from autogluon.tabular import TabularPredictor
    import pandas as pd
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_hastie(n_samples, seed=42):
    """Hastie et al. ESL 10.2: y=1 if sum(x_i^2) > chi2_median(10)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 10)
    chi2_median = 9.3418  # median of chi2(df=10)
    y = (np.sum(X ** 2, axis=1) > chi2_median).astype(int)
    return X, y


def generate_datasets(n_samples, seed=42):
    """Generate all benchmark datasets at given size."""
    datasets = {}

    # Regression
    X, y = make_friedman1(n_samples=n_samples, noise=1.0, random_state=seed)
    datasets['friedman1'] = ('regression', X, y)

    X, y = make_friedman2(n_samples=n_samples, noise=1.0, random_state=seed)
    datasets['friedman2'] = ('regression', X, y)

    X, y = make_friedman3(n_samples=n_samples, noise=1.0, random_state=seed)
    datasets['friedman3'] = ('regression', X, y)

    # Classification
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=seed)
    datasets['moons'] = ('classification', X, y)

    X, y = make_hastie(n_samples=n_samples, seed=seed)
    datasets['hastie'] = ('classification', X, y)

    return datasets


# ---------------------------------------------------------------------------
# wlearn model specs
# ---------------------------------------------------------------------------

REG_MODELS = [
    ('xgb', XGBModel, {'objective': 'reg:squarederror', 'numRound': 100}),
    ('linear', LinearModel, {'solver': 11, 'C': 1.0}),
    ('svm', SVMModel, {'svmType': 3, 'kernel': 2, 'C': 1.0, 'gamma': 0}),
    ('knn', KNNModel, {'k': 5, 'task': 'regression'}),
    ('ebm', EBMModel, {'objective': 'regression'}),
]

CLS_MODELS = [
    ('xgb', XGBModel, {'objective': 'multi:softprob', 'numRound': 100}),
    ('linear', LinearModel, {'solver': 0, 'C': 1.0}),
    ('svm', SVMModel, {'svmType': 0, 'kernel': 2, 'C': 1.0, 'gamma': 0}),
    ('knn', KNNModel, {'k': 5, 'task': 'classification'}),
    ('ebm', EBMModel, {'objective': 'classification'}),
]


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_wlearn(models, X_train, y_train, X_test, y_test, task, strategy, scoring):
    """Run wlearn auto_fit and return (score, time_seconds)."""
    t0 = time.time()
    result = auto_fit(
        models, X_train, y_train,
        strategy=strategy, cv=5, seed=42, task=task, n_iter=20,
    )
    elapsed = time.time() - t0
    model = result['model']
    if model is None:
        return None, elapsed

    if task == 'classification':
        preds = model.predict(X_test)
        score = float(np.mean(preds == y_test))
    else:
        preds = model.predict(X_test)
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    model.dispose()
    return score, elapsed


def run_tpot(X_train, y_train, X_test, y_test, task):
    """Run TPOT and return (score, time_seconds)."""
    if not HAS_TPOT:
        return None, 0

    t0 = time.time()
    try:
        if task == 'classification':
            tpot = TPOTClassifier(
                max_time_mins=1, max_eval_time_mins=0.5,
                cv=5, n_jobs=1, verbose=0,
            )
        else:
            tpot = TPOTRegressor(
                max_time_mins=1, max_eval_time_mins=0.5,
                cv=5, n_jobs=1, verbose=0,
            )

        tpot.fit(X_train, y_train)
        preds = tpot.predict(X_test)
        elapsed = time.time() - t0

        if task == 'classification':
            score = float(np.mean(preds == y_test))
        else:
            ss_res = np.sum((y_test - preds) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return score, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  TPOT error: {e}', flush=True)
        return None, elapsed


def run_autogluon(X_train, y_train, X_test, y_test, task):
    """Run AutoGluon and return (score, time_seconds)."""
    if not HAS_AUTOGLUON:
        return None, 0

    import tempfile
    import shutil

    # AutoGluon needs DataFrames
    cols = [f'f{i}' for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test, columns=cols)

    tmpdir = tempfile.mkdtemp()
    try:
        t0 = time.time()
        predictor = TabularPredictor(
            label='target',
            path=tmpdir,
            problem_type='binary' if task == 'classification' else 'regression',
            verbosity=0,
        )
        predictor.fit(train_df, time_limit=60, presets='medium_quality')
        preds = predictor.predict(test_df).values
        elapsed = time.time() - t0

        if task == 'classification':
            score = float(np.mean(preds == y_test))
        else:
            ss_res = np.sum((y_test - preds) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception as e:
        print(f'  AutoGluon error: {e}', flush=True)
        score = None
        elapsed = 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return score, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_score(score):
    if score is None:
        return '  --  '
    return f'{score:6.4f}'


def fmt_time(t):
    if t == 0:
        return '  -- '
    if t < 1:
        return f'{t*1000:4.0f}ms'
    return f'{t:5.1f}s'


def main():
    sizes = [500, 2000, 10000]

    # Header
    systems = ['wlearn-portfolio', 'wlearn-random']
    if HAS_TPOT:
        systems.append('TPOT')
    if HAS_AUTOGLUON:
        systems.append('AutoGluon')

    header_parts = ['| Dataset   |     n |']
    sep_parts = ['|-----------|------:|']
    for s in systems:
        header_parts.append(f' {s:>16s} |  time |')
        sep_parts.append(f' {"-"*16}:|------:|')

    print(''.join(header_parts))
    print(''.join(sep_parts))

    for n in sizes:
        datasets = generate_datasets(n, seed=42)

        for name, (task, X, y) in datasets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )
            scoring = 'accuracy' if task == 'classification' else 'r2'
            models = CLS_MODELS if task == 'classification' else REG_MODELS

            row = f'| {name:9s} | {n:5d} |'

            # wlearn portfolio
            s, t = run_wlearn(models, X_train, y_train, X_test, y_test,
                              task, 'portfolio', scoring)
            row += f' {fmt_score(s):>16s} | {fmt_time(t):>5s} |'

            # wlearn random
            s, t = run_wlearn(models, X_train, y_train, X_test, y_test,
                              task, 'random', scoring)
            row += f' {fmt_score(s):>16s} | {fmt_time(t):>5s} |'

            # TPOT
            if HAS_TPOT:
                s, t = run_tpot(X_train, y_train, X_test, y_test, task)
                row += f' {fmt_score(s):>16s} | {fmt_time(t):>5s} |'

            # AutoGluon
            if HAS_AUTOGLUON:
                s, t = run_autogluon(X_train, y_train, X_test, y_test, task)
                row += f' {fmt_score(s):>16s} | {fmt_time(t):>5s} |'

            print(row)


if __name__ == '__main__':
    print()
    print('AutoML Benchmark: wlearn vs competitors')
    print(f'TPOT: {"available" if HAS_TPOT else "not installed"}')
    print(f'AutoGluon: {"available" if HAS_AUTOGLUON else "not installed"}')
    print()
    main()
    print()
