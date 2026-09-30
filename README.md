# wlearn

Classical machine learning that runs entirely in the browser and Node.js. No server, no Python runtime, no data leaving your machine.

wlearn packages C/C++ and ONNX-backed models behind a unified, sklearn-style
JavaScript API. Train locally, serialize to a portable WLRN bundle, and use the
same artifact in JavaScript or Python when the corresponding loader is available.

## Why

Most ML libraries require Python and a server. That means network round-trips, data privacy concerns, and infrastructure to manage. For many use cases -- on-device inference, privacy-sensitive data, offline apps, rapid prototyping -- you just want the model to run where the data already is.

WebAssembly makes this possible. The LIBSVM and LIBLINEAR sources used by native
Python tooling can also compile to WASM and run locally in supported browsers and
Node.js. wlearn packages the resulting modules behind a JavaScript API, so model
backends can be installed independently from npm.

## How it works

**Small backend boundaries.** Port packages compile pinned upstream C/C++ source
to WebAssembly through Emscripten; original wlearn packages keep their C11 source
in their own repositories, and Mitra uses ONNX Runtime. Cross-runtime numerical
equivalence is checked per backend with explicit tolerances rather than assumed
from the shared wrapper.

**Unified API.** Models use async construction (WASM must load). Fits are synchronous except Sym family search with Polygrad; `save` is always synchronous; prediction may be sync or async by backend. Pipelines preserve synchronous fit for synchronous children and Promise-lift asynchronous children; ensemble fit is asynchronous because ensembles construct and train owned children.

**Portable bundles.** `save()` produces a self-describing binary bundle (format:
WLRN v1) containing model artifacts, parameters, and a type identifier. `load()`
reads the bundle and dispatches to a registered loader. WLRN is language-neutral;
the Python package implements loaders for the backends listed in its documentation.

**Deterministic cleanup when needed.** WASM models expose `dispose()` for long-running apps, workers, cross-validation, and AutoML loops that create many models.

## Quick start

```
npm install @wlearn/liblinear
```

```js
const { readFileSync, writeFileSync } = require('fs')
const { LinearModel } = require('@wlearn/liblinear')

async function main() {
  // Construction is async because it loads WASM; base-model fit is synchronous.
  const model = await LinearModel.create({
    task: 'classification',
    solver: 'L2R_LR',
    C: 1.0
  })

  const X = [[-2, -2], [-1, -1], [1, 1], [2, 2]]
  const y = new Int32Array([0, 0, 1, 1])
  model.fit(X, y)

  const XTest = [[-1.5, -1.5], [1.5, 1.5]]
  console.log(Array.from(model.predict(XTest))) // [0, 1]

  writeFileSync('linear.wlrn', model.save())
  const restored = await LinearModel.load(readFileSync('linear.wlrn'))
  console.log(Array.from(restored.predict(XTest))) // [0, 1]
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
```

### If you know scikit-learn

The estimator lifecycle is deliberately familiar: create, `fit`, `predict` or
`transform`, `score`, and compose fitted steps in a `Pipeline`. The important
differences are:

| scikit-learn expectation | wlearn contract |
|--------------------------|-----------------|
| Constructors are synchronous | JavaScript WASM models use `await Model.create(params)`; Python construction is synchronous. |
| pandas and NumPy inputs | JavaScript core accepts `number[][]` or a row-major dense typed matrix; individual models may declare sparse CSR support. Python accepts NumPy-compatible arrays. DataFrames are not the core interchange type. |
| `predict_proba()` returns a 2-D array | JavaScript `predictProba()` and Python `predict_proba()` return a flat row-major buffer of `rows * nClasses`; use the fitted `classes` order. |
| pickle/joblib persistence | `save()` writes portable WLRN bytes; import/register the relevant model package before generic `load()`. |
| Python owns native objects through GC | Call `dispose()` in long-running JavaScript loops that create many WASM models. |

Pass `task: 'classification'` or `task: 'regression'` when it is known instead
of relying on label-based inference. Parameters are plain objects and Pipeline
steps receive created estimator instances; wlearn does not implement sklearn's
`step__parameter` convention.

### Preprocessing and Tranfi

wlearn owns estimators, Pipelines, AutoML, and WLRN artifacts. Tranfi is an
independent streaming/data-processing engine. Use `Preprocessor` from
`@wlearn/preprocess` (or `wlearn.preprocess` in Python) for sklearn-style fitted
imputation, encoding, and scaling inside wlearn Pipelines. It stores Tranfi's
immutable learned plan inside a WLRN bundle. Use Tranfi directly for lower-level
byte-stream ETL or typed-batch integrations; it processes streams and batches
rather than exposing a pandas-style in-memory DataFrame.

## Packages

### Core

| Package | Description |
|---------|-------------|
| `@wlearn/types` | TypeScript interfaces plus a minimal runtime constants module. |
| `@wlearn/core` | Matrix helpers, bundle encode/decode, loader registry, pipeline, error classes. Small, no WASM. |
| `@wlearn/preprocess` | Fitted tabular preprocessing adapter over Tranfi, with WLRN persistence. |
| `@wlearn/ensemble` | Stacking, voting, and bagging ensembles. |
| `@wlearn/automl` | Automated model selection with `autoFit()`. Requires model packages. |
| `@wlearn/sdk` | Convenience barrel for Node.js. Re-exports all model classes + core + automl + ensemble. |

### Model ports

| Package | Upstream | What it does |
|---------|----------|--------------|
| `@wlearn/liblinear` | LIBLINEAR v2.50 | Linear SVM and logistic regression. Fast on large sparse datasets. |
| `@wlearn/libsvm` | LIBSVM v3.37 | Kernel SVM (RBF, polynomial, sigmoid). Classification, regression, one-class novelty detection. |
| `@wlearn/xgboost` | XGBoost v3.2.0 | Gradient-boosted trees and random forests for classification and regression. |
| `@wlearn/lightgbm` | LightGBM | Gradient-boosted trees, fast histogram-based. Classification, regression. |
| `@wlearn/nanoflann` | nanoflann v1.6.3 | k-nearest neighbors via KD-trees. Classification and regression. |
| `@wlearn/ebm` | InterpretML v0.7.5 | Explainable boosting machines (GAM). Per-feature shape functions with interpretability. |
| `@wlearn/xlearn` | xLearn v0.44 | Factorization machines (LR, FM, FFM). Tuned for sparse CTR/recommender data. |
| `@wlearn/stochtree` | StochTree | Bayesian additive regression trees (BART). Uncertainty-aware predictions. |
| `@wlearn/tsetlin` | TMU | Tsetlin machine. Interpretable propositional logic classifier. |
| `@wlearn/mitra` | Mitra Tab2D | Pretrained ONNX tabular model with in-context support rows. |

### Native implementations

Built from scratch (not WASM ports of existing libraries):

| Package | Backend | What it does |
|---------|---------|--------------|
| `@wlearn/rf` | C11 | Random forest, ExtraTrees, linear leaves, Hellinger/entropy criteria, pruning, OOB weighting. |
| `@wlearn/nn` | polygrad (C11) | Neural tabular models: MLP, TabM (BatchEnsemble), NAM (Neural Additive Models). |

## API overview

Every model package exports a model class that implements the same contract.
The blocks in this reference section are focused fragments: they assume the
shown model classes plus `X`/`y` are already defined inside an async function.
Use the Quick start above for a complete executable CommonJS program.

### Construction

WASM modules load asynchronously. Use the static `create()` factory:

```js
const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
```

After construction, base-model `fit` and `save` are synchronous. `predict`, `predictProba`, and `score` are synchronous for WASM-backed models but return Promises for async backends (for example, `@wlearn/mitra` uses ONNX Runtime). Pipeline fit remains synchronous with synchronous children and Promise-lifts an asynchronous child. Ensemble fit is asynchronous because ensembles construct and train owned children; use `await composite.fit(X, y)` in code that accepts either kind of composite.

### fit / predict / score

For a WASM-backed base model, these calls are synchronous:

```js
// X: number[][] or { data: Float64Array, rows, cols }
// y: number[] or Int32Array/Float32Array/Float64Array
model.fit(X, y)

const preds = model.predict(X)      // Labels typed array (model-specific)
const accuracy = model.score(X, y)  // accuracy or R-squared
```

### Probability estimates

```js
// liblinear: automatic for logistic regression solvers
const model = await LinearModel.create({ solver: 'L2R_LR' })
model.fit(X, y)
const linearProbs = model.predictProba(X)  // Float64Array, rows * nClasses

// libsvm: set probability: 1
const svm = await SVMModel.create({ svmType: 'C_SVC', kernel: 'RBF', probability: 1 })
svm.fit(X, y)
const svmProbs = svm.predictProba(X)
```

### Save and load

Every model serializes to a WLRN bundle -- a compact binary format with embedded metadata:

```js
const { readFileSync, writeFileSync } = require('fs')

writeFileSync('model.wlrn', model.save())

// Load directly
const directRestored = await LinearModel.load(readFileSync('model.wlrn'))

// Or use the universal loader (auto-dispatches by typeId)
// Importing the model package above registered its loaders.
const { load } = require('@wlearn/core')
const genericRestored = await load(readFileSync('model.wlrn'))

// Bytes are the API representation when you need to store the bundle yourself.
const bytes = model.save()  // Uint8Array
```

The universal `load()` reads the bundle header, finds the registered loader for
that type, and returns a fitted estimator. In a fresh process, first import the
matching model package (or call its explicit registration function); the core
does not eagerly load every optional backend. Nested bundles declare their
required loaders and fail with an actionable error when one is missing.

### Pipeline

Compose multiple steps into a single estimator. Steps are `[name, estimator]` tuples.

```js
const { readFileSync, writeFileSync } = require('fs')
const { Pipeline, load } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')

const model = await LinearModel.create({ task: 'classification' })
const pipe = new Pipeline([['clf', model]])

pipe.fit(X, y)
const preds = pipe.predict(X)

// Save/load works the same as individual models
writeFileSync('pipeline.wlrn', pipe.save())
const restored = await load(readFileSync('pipeline.wlrn'))
restored.predict(X)
```

### Parameters

```js
const params = model.getParams()     // { solver: 'L2R_LR', C: 1.0, ... }
model.setParams({ C: 10.0 })        // update before next fit()

// For AutoML: each model defines its search space
const space = LinearModel.defaultSearchSpace()
// { solver: { type: 'categorical', values: [...] }, C: { type: 'log_uniform', ... }, ... }
```

### Resource lifecycle

```js
model.dispose()
```

`dispose()` releases native/WASM memory immediately. Use it in long-running browser or Node apps, workers, cross-validation, AutoML, and benchmarks where many models are created and discarded. It is not part of the ordinary fit/predict/save path for small scripts.

## Model-specific features

### @wlearn/liblinear

Linear classifiers and regressors. Best for high-dimensional or sparse data where a linear decision boundary suffices.

```js
const { LinearModel, Solver } = require('@wlearn/liblinear')

// Classification with logistic regression
const clf = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
clf.fit(X, y)
clf.predict(X)
clf.predictProba(X)  // probability estimates (LR solvers only)
clf.score(X, y)      // accuracy

// Regression with support vector regression
const reg = await LinearModel.create({ solver: 'L2R_L2LOSS_SVR_DUAL', C: 1.0, p: 0.1 })
reg.fit(X, y)
reg.predict(X)
reg.score(X, y)      // R-squared

// Inspection
clf.nrClass     // 2
clf.nrFeature   // number of features
clf.classes     // Int32Array of class labels
clf.capabilities // { classifier: true, regressor: false, predictProba: true, ... }
```

**Solvers:** `L2R_LR`, `L2R_L2LOSS_SVC_DUAL`, `L2R_L2LOSS_SVC`, `L2R_L1LOSS_SVC_DUAL`, `MCSVM_CS`, `L1R_L2LOSS_SVC`, `L1R_LR`, `L2R_LR_DUAL`, `L2R_L2LOSS_SVR`, `L2R_L2LOSS_SVR_DUAL`, `L2R_L1LOSS_SVR_DUAL`

### @wlearn/libsvm

Kernel SVM for nonlinear classification, regression, and novelty detection.

```js
const { SVMModel, SVMType, Kernel } = require('@wlearn/libsvm')

// Nonlinear classification with RBF kernel
const clf = await SVMModel.create({
  svmType: 'C_SVC',
  kernel: 'RBF',
  C: 10.0,
  gamma: 0.5
})
clf.fit(X, y)
clf.predict(X)
clf.decisionFunction(X)  // signed distances from hyperplane

// Probability estimates (must set probability: 1)
const clf2 = await SVMModel.create({
  svmType: 'C_SVC',
  kernel: 'RBF',
  probability: 1
})
clf2.fit(X, y)
clf2.predictProba(X)

// Regression
const reg = await SVMModel.create({
  svmType: 'EPSILON_SVR',
  kernel: 'RBF',
  C: 10.0,
  gamma: 0.1,
  p: 0.1
})
reg.fit(X, y)
reg.score(X, y)  // R-squared

// One-class SVM (novelty detection)
const oc = await SVMModel.create({
  svmType: 'ONE_CLASS',
  kernel: 'RBF',
  nu: 0.1,
  gamma: 0.5
})
oc.fit(normalData, dummyLabels)
oc.predict(testData)  // +1 (inlier) or -1 (outlier)

// Inspection
clf.nrClass    // number of classes
clf.svCount    // number of support vectors
clf.classes    // Int32Array of class labels
```

**SVM types:** `C_SVC`, `NU_SVC`, `ONE_CLASS`, `EPSILON_SVR`, `NU_SVR`

**Kernels:** `LINEAR`, `POLY`, `RBF`, `SIGMOID`

**Key parameters:** `C` (regularization), `gamma` (kernel width, 0 = auto 1/n_features), `degree` (polynomial), `coef0` (polynomial/sigmoid), `nu` (NU_SVC/NU_SVR), `p` (epsilon-tube width for SVR)

### @wlearn/xgboost

Gradient-boosted trees for classification and regression. Includes random forest mode.

```js
const { XGBModel } = require('@wlearn/xgboost')

// Binary classification
const clf = await XGBModel.create({
  objective: 'binary:logistic',
  max_depth: 6,
  eta: 0.3,
  numRound: 100
})
clf.fit(X, y)
clf.predict(X)        // class labels (0 or 1)
clf.predictProba(X)   // probabilities, shape: rows * 2

// Multiclass
const mc = await XGBModel.create({
  objective: 'multi:softprob',
  num_class: 3,
  numRound: 50
})

// Regression
const reg = await XGBModel.create({
  objective: 'reg:squarederror',
  numRound: 100
})
reg.fit(X, y)
reg.predict(X)
reg.score(X, y)  // R-squared

// Random forest mode
const rf = await XGBModel.create({
  objective: 'binary:logistic',
  numRound: 100,
  num_parallel_tree: 10,
  subsample: 0.8,
  colsample_bynode: 0.8
})
```

**Tested high-level objectives:** `binary:logistic`, `multi:softprob`,
`multi:softmax`, and `reg:squarederror`. Ranking and survival remain low-level
`Booster` tasks because the unified estimator does not yet define their group,
label, and metric contracts.

**Key parameters:** `max_depth`, `eta` (learning rate), `numRound` (number of boosting rounds), `subsample`, `colsample_bytree`, `lambda` (L2 reg), `alpha` (L1 reg), `num_parallel_tree` (for RF mode)

### @wlearn/lightgbm

Gradient-boosted trees with histogram-based learning. Fast training on large datasets.

```js
const { LGBModel } = require('@wlearn/lightgbm')

const clf = await LGBModel.create({
  objective: 'binary',
  num_leaves: 31,
  learning_rate: 0.1,
  numRound: 100
})
clf.fit(X, y)
clf.predict(X)
clf.predictProba(X)
```

### @wlearn/nanoflann

k-nearest neighbors via KD-trees. Fast exact neighbor search for classification and regression.

```js
const { KNNModel } = require('@wlearn/nanoflann')

// Classification
const clf = await KNNModel.create({ k: 5, metric: 'l2', task: 'classification' })
clf.fit(X, y)
clf.predict(X)        // class labels (majority vote among k neighbors)
clf.predictProba(X)   // class proportions, shape: rows * nClasses
clf.score(X, y)       // accuracy

// Regression
const reg = await KNNModel.create({ k: 5, metric: 'l2', task: 'regression' })
reg.fit(X, y)
reg.predict(X)        // mean of k neighbor values
reg.score(X, y)       // R-squared

// Raw neighbor search
const { indices, distances, k: kUsed } = clf.kneighbors(X, 3)
```

**Parameters:** `k` (number of neighbors, default 5), `metric` (`'l2'` or `'l1'`), `leafMaxSize` (KD-tree leaf size, default 10), `task` (`'classification'` or `'regression'`)

### @wlearn/ebm

Explainable boosting machines -- interpretable GAMs with per-feature shape functions.

```js
const { EBMModel } = require('@wlearn/ebm')

const model = await EBMModel.create({ maxRounds: 500, seed: 42 })
model.fit(X, y)

// Standard predict/score
model.predict(X)
model.predictProba(X)

// Explainability
const expl = model.explain(X)          // per-sample, per-term additive contributions
const imp = model.featureImportances() // mean absolute score per term
const shape = model.getShapeFunction(0) // { x, y } for plotting
```

### @wlearn/xlearn

Factorization machines for sparse/CTR data. LR, FM, and FFM with CSR sparse input support.

```js
const { XLearnFMClassifier, XLearnFFMClassifier } = require('@wlearn/xlearn')

// FM classifier
const fm = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
fm.fit(X, y)
fm.predict(X)
fm.predictProba(X)

// FFM with field mapping
const featureFields = new Int32Array([0, 0, 1, 1])
const ffm = await XLearnFFMClassifier.create({ epoch: 10, k: 4, featureFields })
ffm.fit(X, y)

// CSR sparse input
const csr = { rows, cols, data: Float64Array, indices: Int32Array, indptr: Int32Array }
fm.fit(csr, y)
```

Six classes: `XLearnLRClassifier`, `XLearnLRRegressor`, `XLearnFMClassifier`, `XLearnFMRegressor`, `XLearnFFMClassifier`, `XLearnFFMRegressor`.

### @wlearn/stochtree

Bayesian additive regression trees (BART). Uncertainty-aware ensemble of shallow trees.

```js
const { BARTModel } = require('@wlearn/stochtree')

const model = await BARTModel.create({ numTrees: 200, numBurnin: 100, numSamples: 50 })
model.fit(X, y)
model.predict(X)
model.score(X, y)
```

### @wlearn/tsetlin

Tsetlin machine. Interpretable propositional logic classifier using automata-based learning.

```js
const { TsetlinModel } = require('@wlearn/tsetlin')

const model = await TsetlinModel.create({ numClauses: 100, T: 10, s: 3.0 })
model.fit(X, y)
model.predict(X)
```

### @wlearn/mitra

Pretrained Mitra Tab2D models for tabular data. Unlike the generic
`Model.create(params)` form, Mitra construction requires ONNX model bytes or a
pre-created ONNX Runtime session as its first argument. `fit()` synchronously
stores support rows used as in-context examples; prediction is asynchronous.

```js
const { MitraClassifier, MitraRegressor } = require('@wlearn/mitra')
const ort = require('onnxruntime-node')

// Classification
const clfSession = await ort.InferenceSession.create('mitra-classifier.onnx')
const clf = await MitraClassifier.create(clfSession, { maxSupport: 50 }, { ort })
clf.fit(X, y)
const preds = await clf.predict(Xtest)  // async (ONNX inference)

// Regression
const regSession = await ort.InferenceSession.create('mitra-regressor.onnx')
const reg = await MitraRegressor.create(regSession, { maxSupport: 50 }, { ort })
reg.fit(X, y)
const rPreds = await reg.predict(Xtest)
```

Requires `onnxruntime-node` (Node.js) or `onnxruntime-web` (browser) as peer dependency. ONNX model files must be downloaded separately (see package README).

### @wlearn/nn

Neural tabular models powered by [polygrad](https://github.com/polygrad/polygrad) (C11 tensor framework). Three architectures: MLP, TabM (BatchEnsemble), and NAM (Neural Additive Models).

```js
const { MLPClassifier, TabMClassifier, NAMClassifier } = require('@wlearn/nn')

// MLP -- standard multilayer perceptron
const mlp = await MLPClassifier.create({
  hidden_sizes: [64, 32], activation: 'relu', epochs: 100, lr: 0.01,
  optimizer: 'adam', batch_size: 32, seed: 42
})
mlp.fit(X, y)
mlp.predict(X)
mlp.score(X, y)

// TabM -- BatchEnsemble MLP (SOTA on tabular benchmarks)
const tabm = await TabMClassifier.create({
  hidden_sizes: [64, 32], n_ensemble: 32, activation: 'relu',
  epochs: 100, lr: 0.01, optimizer: 'adam', seed: 42
})
tabm.fit(X, y)
tabm.predict(X)

// NAM -- Neural Additive Model (interpretable)
const nam = await NAMClassifier.create({
  hidden_sizes: [64, 64], activation: 'exu', epochs: 200, lr: 0.001,
  optimizer: 'adam', seed: 42
})
nam.fit(X, y)
nam.predict(X)
```

**MLP** is a standard feedforward network. Supports mini-batch training, early stopping, and multiple activations (relu, gelu, silu).

**TabM** (Gorishniy et al., 2024) adds per-layer BatchEnsemble adapters to an MLP. Each ensemble member i applies rank-1 perturbations: `l_i(x) = s_i * (W @ (r_i * x)) + b_i`. Predictions are averaged over k members. Best average rank across 46 tabular datasets, beating XGBoost and CatBoost.

**NAM** (Agarwal et al., 2021) is a neural additive model: `g(E[y]) = beta + f1(x1) + ... + fK(xK)`. Each f_k is a small MLP on a single feature. Interpretable per-feature shape functions. Supports ExU activation (Exponential Unit) for sharp function learning.

All three support classification and regression, save/load via WLRN bundles, and share the same Estimator API.

### @wlearn/ensemble

Ensemble methods that combine multiple models for better predictions.

```js
const { StackingEnsemble, VotingEnsemble, BaggedEstimator } = require('@wlearn/ensemble')
```

`StackingEnsemble` trains base models with out-of-fold predictions and feeds them to a meta-learner. `VotingEnsemble` averages predictions (soft vote) or takes majority class (hard vote). `BaggedEstimator` trains multiple copies of a single model over repeated K-fold splits.

### @wlearn/automl

Automated model selection: searches hyperparameter spaces across multiple model families, selects the best via cross-validation, and optionally builds an ensemble.

```js
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')
const { XGBModel } = require('@wlearn/xgboost')

const models = [
  { name: 'linear', classId: 'wlearn.liblinear.classifier@1',
    portfolioKey: 'linear', cls: LinearModel, params: { task: 'classification' } },
  { name: 'xgb', classId: 'wlearn.xgboost.classifier@1',
    portfolioKey: 'xgb', cls: XGBModel, params: { task: 'classification' } }
]

const result = await autoFit(models, X, y, {
  strategy: 'random',    // 'random' | 'halving' | 'portfolio' | 'progressive' | 'bayesian'
  ensemble: true,         // build Caruana ensemble from top candidates
  ensembleSize: 20,
  refit: true,            // refit best model on full data
  onProgress: ({ phase, progress }) => console.log(phase, progress)
})

result.model           // best fitted estimator (or ensemble)
result.leaderboard     // ranked candidate results
result.archive         // structured Archive of ok/failed trials
result.bestScore       // best CV score
result.bestModelName   // e.g. 'xgb'
result.bestParams      // winning hyperparameters
```

For the simple path, stop there: use `result.model` to predict, `result.leaderboard` to inspect candidates, and ignore `result.archive` unless you need run provenance, failed-candidate inspection, or an agent-readable ledger.

`@wlearn/automl` requires at least one model package (e.g. `@wlearn/xgboost`) to do anything useful.

### Structured task/prediction/archive API

`@wlearn/core` also exposes structured primitives for apps and agents. These are optional; they are not required to train a model, run `autoFit()`, save a `.wlrn` bundle, or make predictions.

- `createTask()` records dataset shape, labels, groups, row roles, feature schema, and provenance.
- `createPrediction()` records responses/probabilities with explicit class order.
- `listMeasures()` and `evaluateMetricSet()` expose metric names, directions, sample-weight support, multiclass AUC, and undefined-metric handling without guessing.
- `createResamplingPlan()` creates deterministic holdout/k-fold/group/time-series plus sliding row/index/period splits.
- `Archive` records candidate params, scores, timings, failed runs, and leaderboards.

See the structured API sections in [@wlearn/core](js/core/README.md) and
[Python wlearn](py/README.md) for the concrete contracts.

## Python

The WLRN container is shared by JavaScript and Python. For backend pairs that
have corresponding loaders in both runtimes, a bundle written in one can be
loaded in the other and checked within that backend's declared tolerance.

```python
import wlearn.xgboost  # registers loader

# Load a bundle (produced by JS or Python)
model = wlearn.load('model.wlrn')
preds = model.predict(X)
model.score(X, y)

# Save back to WLRN (loadable from JS)
model.save('model-resaved.wlrn')
```

The Python package depends on NumPy. Wrappers exist for: xgboost, liblinear, libsvm, nanoflann, lightgbm, ebm, xlearn, stochtree, tsetlin, nn. Classical ML wrappers use native upstream packages where training needs them; xlearn and EBM bundle inference are NumPy-only. Neural models (nn) use polygrad via ctypes.

```
pip install wlearn               # core, metrics, resampling, AutoML/ensemble primitives; no optional training backend
pip install wlearn[xgboost]      # + xgboost support
pip install wlearn[liblinear]    # + liblinear support
pip install wlearn[libsvm]       # + libsvm support
pip install wlearn[nanoflann]    # + k-nearest neighbors support
pip install wlearn[lightgbm]     # + LightGBM support
pip install wlearn[stochtree]    # + BART support
pip install wlearn[nn]           # + polygrad neural models
pip install wlearn[preprocess]   # + Tranfi-backed fitted preprocessing
pip install wlearn[bo]           # + Bayesian AutoML strategy support
pip install wlearn[all]          # everything
```

Requires Python 3.10+.

## Testing

Use the Makefile for repo-level checks:

```bash
make test          # JS workspaces + focused Python core/AutoML suite
make test-browser  # rebuild browser bundles, then run Playwright smoke tests
make test-py       # full Python suite; requires optional backend deps
make test-z3       # optional Z3 proof smoke for resampling index arithmetic
```

`npm test` delegates to `make test`; `npm run test:js` runs JavaScript only.
`npm run test:py-core` delegates to the same focused Python check as Make,
including nested-bundle hardening. Select Python with `WLEARN_PYTHON` or Make's
`PYTHON` override. `npm run test:all` delegates to `make test-all`, including
ecosystem composition fixtures. Browser checks reuse this repo's Playwright
install and Chromium cache; set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` when needed.
`make test-z3` requires `z3-solver<4.15.4` and is optional.

For multi-repo development, keep package manifests on real semver dependencies and overlay local sibling repos with:

```bash
npm run dev:link        # symlink sibling @wlearn packages into node_modules/@wlearn
npm run dev:link:force  # replace installed package dirs with local symlinks
npm run dev:unlink      # remove symlinks created by the linker
npm run dev:check       # check local dependency ranges and installed core identity
```

`dev:link` does not edit `package.json`; publishing metadata remains the same as a registry install.
The linker uses only Node built-ins, so it can be run before `npm install` when testing unreleased sibling package versions.
`dev:check` reads local source versions and resolves installed core paths without
importing models, changing links or contacting registries. It detects stale pins
even when source symlinks mask them. Isolated packed consumers remain the final
check of installation behavior.

## Cross-language interop

For model/backend pairs covered by the golden fixtures, the interoperability
tests require:

- **Identical blob bytes**: upstream serialization produces the same bytes regardless of host language
- **Equivalent predictions**: models loaded from the same bundle agree within the fixture's declared floating-point tolerance
- **Round-trip safe**: JS -> Python -> JS preserves model bytes exactly

Golden fixture tests verify all three directions:
- `fixtures/verify.mjs` validates JS-produced bundles
- `py/tests/test_compat.py` loads JS fixtures in Python, verifies format and predictions
- `fixtures/verify-py-bundles.mjs` validates Python-produced bundles back in JS

The fixture generators and verifiers define an explicit expected set. Before a
release, every expected bundle and sidecar must be tracked and
`npm run test:interop` must pass with every Python backend installed and no skipped
fixture. `npm run test:interop:minimal` is the explicit developer lane for a partial
backend environment; its output index records every skipped backend and must never
be reported as full interoperability. The combined runner creates a fresh output
directory and prints its path. Set `WLEARN_INTEROP_OUTPUT_DIR` to an empty directory
to retain results at a chosen location, and `WLEARN_PYTHON` to select the Python
executable. Standalone Python tests use a temporary directory by default; standalone
reverse verification requires the matching `WLEARN_INTEROP_OUTPUT_DIR`. Reusing a
nonempty output directory fails without deleting it. An exclusive ownership marker
also rejects overlapping writers targeting the same empty directory. The reverse verifier rejects
missing indexes or declared outputs, and prediction checks reject empty, NaN, and
infinite outputs before applying tolerances.

## Bundle format

wlearn uses a compact binary format (WLRN v1) for model persistence. Every bundle is self-describing:

```
[4 bytes]  magic: "WLRN"
[4 bytes]  version: 1
[4 bytes]  manifest length
[4 bytes]  TOC length
[N bytes]  manifest (JSON): { typeId, bundleVersion, params, ... }
[M bytes]  TOC (JSON): [{ id, offset, length, sha256 }, ...]
[... ]     blob data (raw model weights)
```

The `typeId` field (e.g., `wlearn.liblinear.classifier@1`, `wlearn.xgboost.regressor@1`) tells the loader registry which deserializer to use. This makes bundles portable across languages and runtimes.

Canonical v1 requires manifest fields `typeId`, `bundleVersion`, `requires`,
`params`, and `artifacts`. TOC records contain exactly `id`, `offset`, `length`,
`sha256`, and `mediaType`; artifact declarations omit only `offset`. Canonical
writers reject legacy nested bundles, so every writer output passes strict
recursive validation.

Default decoders retain compatibility with historical v1 artifacts that omit
`requires`, `params`, `artifacts`, or TOC `mediaType`, use non-canonical TOC order,
or carry fixed-record extensions. Safety checks—bounds, portable JSON, blob
coverage, hashes, and recursion budgets—still apply. Use
`validateBundle(bytes, { allowLegacyManifest: false })` in JS or
`validate_bundle(data, allow_legacy_manifest=False)` in Python for a canonical
conformance gate. Legacy inputs may lack complete dependency-preflight metadata;
writers never reproduce that shape. This read compatibility remains for the
current major release and can only be removed with a major-version migration and
advance deprecation notice.

```js
const { encodeBundle, decodeBundle } = require('@wlearn/core')

// Encode a bundle
const artifacts = [{ id: 'model', mediaType: 'application/octet-stream', data: modelBytes }]
const manifest = { typeId: 'my.custom.model@1', params: { lr: 0.01 } }
const bundle = encodeBundle(manifest, artifacts)  // Uint8Array

// Decode a bundle
const { manifest, toc, blobs } = decodeBundle(bundle)
console.log(manifest.typeId)    // 'wlearn.liblinear.classifier@1'
console.log(manifest.params)    // { solver: 'L2R_LR', C: 1.0, ... }
console.log(toc[0].id)          // 'model'
console.log(toc[0].sha256)      // hex hash of model blob

// blobs is a single concatenated Uint8Array; slice using toc offsets:
const modelBlob = blobs.slice(toc[0].offset, toc[0].offset + toc[0].length)
```

## Performance tips

**Use typed matrices for large datasets.** Passing `number[][]` to `fit()` or `predict()` triggers a copy into `Float64Array`. For repeated calls or large data, pre-convert:

```js
const X = {
  data: new Float64Array(buffer),  // row-major, contiguous
  rows: 1000,
  cols: 50
}
model.fit(X, y)
```

**Prefer batch prediction.** Model wrappers accept all rows in one matrix, avoiding
a separate public JavaScript call for each row.

**Dispose promptly in loops.** If you are training many models (grid search, cross-validation), dispose each one before creating the next to release WASM heap memory promptly.

## Install

### JavaScript

```
npm install @wlearn/liblinear    # linear SVM + logistic regression
npm install @wlearn/libsvm       # kernel SVM
npm install @wlearn/xgboost      # gradient-boosted trees + random forests
npm install @wlearn/lightgbm     # histogram-based gradient boosting
npm install @wlearn/nanoflann    # k-nearest neighbors (KD-tree)
npm install @wlearn/ebm          # explainable boosting machines
npm install @wlearn/xlearn       # factorization machines (LR/FM/FFM)
npm install @wlearn/stochtree    # BART
npm install @wlearn/tsetlin      # Tsetlin machine
npm install @wlearn/mitra onnxruntime-node   # pretrained tabular models (ONNX, Node.js)
npm install @wlearn/mitra onnxruntime-web    # pretrained tabular models (ONNX, browser)
npm install @wlearn/nn            # neural tabular models (MLP, TabM, NAM)
npm install @wlearn/ensemble     # stacking, voting, bagging
npm install @wlearn/automl       # automated model selection (needs model packages)
npm install @wlearn/preprocess   # fitted tabular preprocessing (Tranfi-backed)
npm install @wlearn/core         # just the core (bundle format, registry, pipeline)
```

Install the Node convenience barrel for its listed model/core package set:

```
npm install @wlearn/sdk
```

`@wlearn/sdk` re-exports its listed model classes, `autoFit`, `Pipeline`, `load`,
metrics, and cross-validation utilities. It does not include the canonical
Tranfi-backed `@wlearn/preprocess`; install and import that package separately. It
also treats `@wlearn/mitra` as optional because ONNX Runtime is a peer dependency.
The SDK is Node/scripting-only; browser users should import individual packages.

Or install packages individually:

```
npm install @wlearn/core @wlearn/preprocess @wlearn/automl @wlearn/ensemble @wlearn/liblinear @wlearn/libsvm @wlearn/xgboost @wlearn/lightgbm @wlearn/nanoflann @wlearn/ebm @wlearn/xlearn @wlearn/stochtree @wlearn/tsetlin @wlearn/mitra onnxruntime-node
```

The JavaScript runtime/model packages use CommonJS entry points. Browser-capable
packages provide their documented browser builds; the SDK is the Node-only
exception.

```js
const { LinearModel } = require('@wlearn/liblinear')
```

## Repository structure

This repository keeps JavaScript workspaces in `js/{types,core,preprocess,ensemble,automl,sdk}`
and the Python distribution in `py/`. Shared integration fixtures live in
`fixtures/`; development commands live in `scripts/` and the root Makefile.
The former `packages/` source paths moved to `js/`; npm package names are unchanged.

| Repo | Package | Description |
|------|---------|-------------|
| [wlearn](https://github.com/wlearn-org/wlearn) | `@wlearn/types`, `@wlearn/core`, `@wlearn/preprocess`, `@wlearn/sdk`, `@wlearn/automl`, `@wlearn/ensemble` | Core monorepo + Python `wlearn` |
| [liblinear-wasm](https://github.com/wlearn-org/liblinear-wasm) | `@wlearn/liblinear` | Linear SVM, logistic regression |
| [libsvm-wasm](https://github.com/wlearn-org/libsvm-wasm) | `@wlearn/libsvm` | Kernel SVM (RBF, poly, sigmoid) |
| [xgboost-wasm](https://github.com/wlearn-org/xgboost-wasm) | `@wlearn/xgboost` | Gradient boosting + RF mode |
| [lightgbm-wasm](https://github.com/wlearn-org/lightgbm-wasm) | `@wlearn/lightgbm` | Histogram boosting |
| [nanoflann-wasm](https://github.com/wlearn-org/nanoflann-wasm) | `@wlearn/nanoflann` | KNN via KD-tree |
| [ebm-wasm](https://github.com/wlearn-org/ebm-wasm) | `@wlearn/ebm` | Explainable boosting machine |
| [xlearn-wasm](https://github.com/wlearn-org/xlearn-wasm) | `@wlearn/xlearn` | Factorization machines (LR/FM/FFM) |
| [stochtree-wasm](https://github.com/wlearn-org/stochtree-wasm) | `@wlearn/stochtree` | BART |
| [tsetlin-wasm](https://github.com/wlearn-org/tsetlin-wasm) | `@wlearn/tsetlin` | Tsetlin machine |
| [mitra-onnx](https://github.com/wlearn-org/mitra-onnx) | `@wlearn/mitra` | Pretrained ONNX tabular models |
| [rf](https://github.com/wlearn-org/rf) | `@wlearn/rf` | Random forest, ExtraTrees (C11) |
| [nn](https://github.com/wlearn-org/nn) | `@wlearn/nn` | MLP, TabM, NAM (polygrad) |
| [gam](https://github.com/wlearn-org/gam) | `@wlearn/gam` | GLM/GAM/Cox (C11) |
| [cluster](https://github.com/wlearn-org/cluster) | `@wlearn/cluster` | K-Means, DBSCAN, hierarchical (C11) |
| [basis](https://github.com/wlearn-org/basis) | `@wlearn/basis` | Fused estimators and independent feature maps (C11) |

Website: [wlearn.org](https://wlearn.org)

WASM port repos carry upstream C/C++ source as git submodules. C11 repos (rf, gam, cluster, bo, basis) are written from scratch with canonical C in root `src/`, JS packages in `js/`, and standalone Python packages in `py/`. Python wrappers for upstream-native packages live in the core repo.

## License

Packages in this core repository are Apache-2.0. Each model repository carries
its own package license, notices, and upstream attribution; consult its `LICENSE`
and `NOTICE` files before redistribution.

## Current execution scope

Pipeline runs sequential steps and AutoML evaluates candidates/folds serially.
DAG execution, TensorRef routing, and a worker scheduler remain planned. Explicit
CV fold arrays and resampling plans are supported for evaluation; OOF/stacking
require complete, non-repeated test coverage. Advanced temporal split generators
are experimental. Probability scoring uses class order and the Measure's declared
optimization direction; uncertainty estimation remains a separate planned effort.

Browser bundles that compose models must use the same exact core version. Core
shares runtime identity within each JavaScript realm and rejects mixed versions.
Rebuild all browser artifacts after a core update.

### Coordinated releases

`scripts/release.py` publishes a qualified ecosystem manifest in dependency order.
It uses the tested npm tarballs and Python sdists, pushes their recorded commits
and tags, and creates GitHub releases with those exact archives. It never builds
from the publishing machine's worktree.

```sh
python3 scripts/release.py check /path/to/release-manifest.json
python3 scripts/release.py publish /path/to/release-manifest.json --create-repos
```

The host needs npm, GitHub and PyPI credentials, plus Git, npm, `gh` and Twine.
Use `--twine '/path/to/python -m twine'` for a separate publishing environment.
`--create-repos` permits creating missing public repositories under `wlearn-org`.

Preparation must record passing checks, archive hashes, committed package
metadata and dependency order in the manifest. The driver checks the whole set
before writing: an existing version, tag or release asset with different content
is an error. Rerunning the same command skips matching uploads and completes
missing release assets. A partial publication is resumable, not transactional.
