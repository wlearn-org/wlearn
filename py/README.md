# wlearn (Python)

Portable ML computation primitives for Python. Train models with native backends, save to cross-language `.wlrn` bundles, load bundles produced by JS `@wlearn/*` packages. The base package depends on NumPy because core task, prediction, measure, resampling, AutoML, and ensemble primitives operate on numeric arrays.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
pip install wlearn
```

Install with model backends as needed:

```bash
pip install wlearn[xgboost]      # XGBoost
pip install wlearn[liblinear]    # Logistic regression, linear SVM
pip install wlearn[libsvm]       # Kernel SVM
pip install wlearn[nanoflann]    # KNN
pip install wlearn[ebm-fit]      # EBM training (interpret)
pip install wlearn[lightgbm]     # LightGBM
pip install wlearn[stochtree]    # BART training (stochtree)
pip install wlearn[tsetlin-fit]  # Tsetlin machine training (tmu)
pip install wlearn[nn]           # Neural tabular models (polygrad)
pip install wlearn[bo]           # Bayesian AutoML strategy (wlearn-bo)
pip install wlearn[all]          # All backends
```

## Quick start

```python
import numpy as np
import wlearn
from wlearn.xgboost import XGBModel

# Train
model = XGBModel.create({'objective': 'binary:logistic', 'max_depth': 3, 'numRound': 50})
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('accuracy:', model.score(X_test, y_test))

# Save to .wlrn file (loadable from JS @wlearn/xgboost too)
model.save('model.wlrn')

# Load from bundle
restored = wlearn.load('model.wlrn')
restored.predict(X_test)
```

## API

### Model wrappers

Each wrapper trains with a native Python backend and produces `.wlrn` bundles compatible with the corresponding JS `@wlearn/*` package.

| Module | Class | Backend | Tasks |
|--------|-------|---------|-------|
| `wlearn.xgboost` | `XGBModel` | xgboost | classification, regression |
| `wlearn.liblinear` | `LinearModel` | liblinear-official | classification, regression |
| `wlearn.libsvm` | `SVMModel` | libsvm-official | classification, regression |
| `wlearn.nanoflann` | `KNNModel` | pynanoflann | classification, regression |
| `wlearn.lightgbm` | `LGBModel` | lightgbm | classification, regression |
| `wlearn.ebm` | `EBMModel` | numpy (inference), interpret (fit) | classification, regression |
| `wlearn.xlearn` | `XLearnModel` | numpy inference for xLearn bundles | classification, regression |
| `wlearn.stochtree` | `BARTModel` | stochtree | classification, regression |
| `wlearn.tsetlin` | `TsetlinModel` | tmu | classification, regression |
| `wlearn.nn` | `MLPClassifier`, `MLPRegressor`, `TabMClassifier`, `TabMRegressor`, `NAMClassifier`, `NAMRegressor` | polygrad (ctypes) | classification, regression |

All models share a common API:

```python
model = Model.create(params)     # create unfitted
model.fit(X, y)                  # train
model.predict(X)                 # predict labels
model.predict_proba(X)           # predict probabilities (classifiers)
model.score(X, y)                # accuracy (clf) or R^2 (reg)
bundle = model.save()            # return .wlrn bytes
model.save('model.wlrn')         # write .wlrn file and return bytes
```

Models that wrap native handles also expose `dispose()` for deterministic cleanup in long-running processes; ordinary scripts can usually rely on Python object cleanup.

### Bundle format

The `.wlrn` bundle is a binary container (header + JSON manifest + TOC + blobs) designed for cross-language compatibility. Bundles produced in Python can be loaded in JS and vice versa.

```python
from wlearn import encode_bundle, decode_bundle, validate_bundle, load

# Low-level encode
data = encode_bundle(
    manifest={'typeId': 'wlearn.xgboost.classifier@1', 'params': {...}},
    artifacts=[{'id': 'model', 'data': model_bytes}]
)

# Decode
manifest, toc, blobs = decode_bundle(data)

# Load via registry (dispatches to correct model wrapper)
model = load(data)
model = load('model.wlrn')
```

Canonical writers always include `bundleVersion`, `requires`, `params`, and an
`artifacts` declaration exactly matching the TOC. Fixed TOC and artifact
declaration records have no extension fields, and nested WLRN artifacts must also
be canonical. Call
`validate_bundle(data, allow_legacy_manifest=False)` to prove conformance.

The default decoder is intentionally a compatibility-ingestion path for older v1
artifacts. It permits missing `requires`, `params`, `artifacts`, and TOC
`mediaType`, non-canonical TOC ordering, and extensions on fixed records. Bounds,
portable JSON, exact blob coverage, hashes, recursive validation, and all present
fields remain enforced. An old bundle without `requires` cannot provide complete
nested-loader preflight. Writers never produce legacy bundles. Compatibility reads
are supported through this major release; removal requires a major release,
migration tooling, and advance notice.

### Registry

Model wrappers register their loaders automatically on import. The `load()` function reads the bundle's `typeId` and dispatches to the registered loader.

```python
from wlearn import register, load

# Custom loader
register('myorg.custom@1', lambda manifest, toc, blobs: MyModel(blobs[0]))
model = load(bundle_bytes)
```

### Pipeline

```python
from wlearn import Pipeline
from wlearn.xgboost import XGBModel
from wlearn.scalers import StandardScaler

scaler = StandardScaler()
model = XGBModel.create({'objective': 'binary:logistic'})
pipe = Pipeline([('scaler', scaler), ('clf', model)])

pipe.fit(X_train, y_train)
pipe.predict(X_test)
pipe.score(X_test, y_test)

# Save/load preserves the full pipeline
pipe.save('pipeline.wlrn')
restored = wlearn.load('pipeline.wlrn')
```

### Task, prediction, measures, resampling, archive

Python exposes the same structured primitives as JS so agents and apps can use one mental model:

```python
from wlearn import (
    create_task, create_prediction, create_resampling_plan,
    evaluate_metric_set, Archive,
)

task = create_task(id='toy', X=X_train, y=y_train)
plan = create_resampling_plan(strategy='stratified_kfold', n=len(y_train), y=y_train, k=5)
pred = create_prediction(truth=y_test, response=response, proba=proba, classes=classes)
scores = evaluate_metric_set(['accuracy', 'log_loss', 'roc_auc_ovr'], pred)

archive = Archive(task_id=task.id, measures=['accuracy'], primary_measure='accuracy')
archive.add({'trial_id': 'xgb-0', 'candidate_id': 'xgb-depth6', 'scores': scores, 'status': 'ok'})
archive.leaderboard()
```

Measures support sample weights, multiclass AUC (`roc_auc_ovr`, `roc_auc_ovo`), and explicit undefined-metric policies. Resampling plans cover holdout/k-fold/group/time-series plus sliding row, index, and period windows for rolling validation.

`wlearn.automl.auto_fit()` returns `archive` as well as `leaderboard`, so failed candidates, params, fold scores, and timings are queryable after a run.

### AutoML preprocessing

Install the optional prepared-transform backend with `pip install wlearn[preprocess]`.
`auto_fit()` then fits a fresh Tranfi-backed preprocessor inside every CV and OOF
training fold, and returns the refitted winner as a `Pipeline`:

```python
from wlearn.automl import auto_fit

result = auto_fit(
    [{'name': 'linear', 'classId': 'wlearn.liblinear.classifier@1',
      'cls': LinearModel}],
    X, y,
    preprocess={'impute': 'auto', 'encode': 'onehot', 'scale': 'standard'},
)

result['bestCandidate']       # structured model + resolved preprocessing identity
result['model'].provenance    # persisted in the Pipeline WLRN artifact
```

`preprocess` also accepts `True` for defaults or a list of fixed templates with
`templateId`, `typeId='wlearn.preprocess.tabular@1'`, and `params`. All search
strategies cross candidates with every template. Candidate IDs are opaque
`wlc1_<sha256>` values; use the structured candidate instead of parsing the ID.
Raw-feature stacking passthrough is rejected when preprocessing is active because
fold-fitted feature spaces may differ.

For the simple path, ignore these primitives: call `Model.create()`, `fit()`, `predict()`, `score()`, and `save()`, or use `auto_fit()` and read `result['model']`, `result['leaderboard']`, and `result['bestScore']`. The archive is there when you need provenance or agent-readable run history.

## Testing

```bash
pip install wlearn[test]
PYTHONPATH=py python -m pytest py/tests/test_ecosystem.py py/tests/test_properties.py py/tests/test_automl.py -q

# Full suite; install optional backend extras first.
PYTHONPATH=py python -m pytest py/tests -q

# Optional solver smoke for resampling interval arithmetic.
pip install 'z3-solver<4.15.4'
PYTHONPATH=py python py/tests/external/resampling_z3.py
```

The focused suite covers the shared primitives, Hypothesis property probes, and AutoML path without optional native backends. The Z3 smoke follows Polygrad's external-check pattern and stays out of the normal unit-test dependency set.

## Cross-language parity

Golden `.wlrn` fixtures in `fixtures/` are generated by JS, then verified in Python (and vice versa). Tests ensure:

- Bundle structure (manifest, TOC, blob SHA-256) matches across languages
- Model predictions match within tolerance (atol=1e-5)
- Bundles saved from Python load correctly in JS

## License

Apache-2.0
