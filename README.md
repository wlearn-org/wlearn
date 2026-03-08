# wlearn

Classical machine learning that runs entirely in the browser and Node.js. No server, no Python runtime, no data leaving your machine.

wlearn compiles battle-tested C/C++ ML libraries to WebAssembly and wraps them in a unified, sklearn-style JavaScript API. Train a model, serialize it to a portable binary bundle, load it anywhere -- same predictions, same format, JS or Python.

## Why

Most ML libraries require Python and a server. That means network round-trips, data privacy concerns, and infrastructure to manage. For many use cases -- on-device inference, privacy-sensitive data, offline apps, rapid prototyping -- you just want the model to run where the data already is.

WebAssembly makes this possible. The same optimized C code that powers scikit-learn's SVM and linear classifiers can compile to WASM and run at near-native speed in any modern browser or Node.js process. wlearn packages these compilers into npm modules with a clean JS API, so you can `npm install` a classifier the same way you `pip install` one.

## How it works

**WASM ports, not reimplementations.** Each model package compiles the original upstream C/C++ source to WebAssembly via Emscripten. The numerical results match the native libraries.

**Unified API.** Every model follows the same pattern: async construction (WASM must load), then synchronous `fit`, `predict`, `score`, `save`, `dispose`. No surprises if you know scikit-learn.

**Portable bundles.** `save()` produces a self-describing binary bundle (format: WLRN v1) containing the model weights, hyperparameters, and a type identifier. `load()` reads the bundle and dispatches to the right loader automatically. Bundles are language-agnostic -- the Python `wlearn` package reads the same files.

**Mandatory resource management.** WASM models allocate linear memory that the JavaScript garbage collector cannot see. Every model has a `dispose()` method. Call it when you are done.

## Quick start

```
npm install @wlearn/liblinear
```

```js
const { LinearModel } = require('@wlearn/liblinear')

// 1. Create (async -- loads WASM)
const model = await LinearModel.create({
  solver: 'L2R_LR',
  C: 1.0
})

// 2. Train (sync)
const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
const y = [0, 0, 1, 1]
model.fit(X, y)

// 3. Predict (sync)
const predictions = model.predict([[2, 3], [6, 7]])
console.log(predictions)  // Float64Array [0, 1]

// 4. Save to portable bundle
const bundle = model.save()  // Uint8Array (WLRN format)

// 5. Load anywhere
const restored = await LinearModel.load(bundle)
restored.predict([[6, 7]])  // same result

// 6. Clean up
model.dispose()
restored.dispose()
```

## Packages

### Core

| Package | Description |
|---------|-------------|
| `@wlearn/types` | TypeScript interfaces and constants. Zero runtime. |
| `@wlearn/core` | Matrix helpers, bundle encode/decode, loader registry, pipeline, error classes. Small, no WASM. |
| `@wlearn/ensemble` | Stacking, voting, and bagging ensembles. |
| `@wlearn/automl` | Automated model selection with `autoFit()`. Requires model packages. |
| `@wlearn/sdk` | Convenience barrel for Node.js. Re-exports all model classes + core + automl + ensemble. |

### Model ports

| Package | Upstream | What it does | Tests |
|---------|----------|--------------|-------|
| `@wlearn/liblinear` | LIBLINEAR v2.50 | Linear SVM and logistic regression. Fast on large sparse datasets. | 23 |
| `@wlearn/libsvm` | LIBSVM v3.37 | Kernel SVM (RBF, polynomial, sigmoid). Classification, regression, one-class novelty detection. | 27 |
| `@wlearn/xgboost` | XGBoost v3.2.0 | Gradient-boosted trees, random forests. Classification, regression, ranking. | 51 |
| `@wlearn/lightgbm` | LightGBM | Gradient-boosted trees, fast histogram-based. Classification, regression. | 34 |
| `@wlearn/nanoflann` | nanoflann v1.6.3 | k-nearest neighbors via KD-trees. Classification and regression. | 27 |
| `@wlearn/ebm` | InterpretML v0.7.5 | Explainable boosting machines (GAM). Per-feature shape functions with interpretability. | 27 |
| `@wlearn/xlearn` | xLearn v0.44 | Factorization machines (LR, FM, FFM). Tuned for sparse CTR/recommender data. | 44 |
| `@wlearn/stochtree` | StochTree | Bayesian additive regression trees (BART). Uncertainty-aware predictions. | 27 |
| `@wlearn/tsetlin` | TMU | Tsetlin machine. Interpretable propositional logic classifier. | 26 |
| `@wlearn/mitra` | Mitra Tab2D | Pretrained ONNX tabular models. Zero-shot and fine-tuned inference via ONNX Runtime. | 26 |

### Native implementations

Built from scratch (not WASM ports of existing libraries):

| Package | Backend | What it does | Tests |
|---------|---------|--------------|-------|
| `@wlearn/rf` | C11 | Random forest, ExtraTrees, linear leaves, Hellinger/entropy criteria, pruning, OOB weighting. | 151 |
| `@wlearn/nn` | polygrad (C11) | Neural tabular models: MLP, TabM (BatchEnsemble), NAM (Neural Additive Models). | 117 |

## API overview

Every model package exports a model class that implements the same contract.

### Construction

WASM modules load asynchronously. Use the static `create()` factory:

```js
const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
```

After construction, `fit`, `save`, and `dispose` are synchronous. `predict`, `predictProba`, and `score` are synchronous for WASM-backed models but return Promises for async backends (e.g. `@wlearn/mitra` uses ONNX Runtime).

### fit / predict / score

```js
// X: number[][] or { data: Float64Array, rows, cols }
// y: number[] or Float64Array
model.fit(X, y)

const preds = model.predict(X)       // Float64Array
const accuracy = model.score(X, y)   // number (accuracy or R-squared)
```

### Probability estimates

```js
// liblinear: automatic for logistic regression solvers
const model = await LinearModel.create({ solver: 'L2R_LR' })
model.fit(X, y)
const probs = model.predictProba(X)  // Float64Array, shape: rows * nClasses

// libsvm: set probability: 1
const svm = await SVMModel.create({ svmType: 'C_SVC', kernel: 'RBF', probability: 1 })
svm.fit(X, y)
const probs = svm.predictProba(X)
```

### Save and load

Every model serializes to a WLRN bundle -- a compact binary format with embedded metadata:

```js
const bytes = model.save()  // Uint8Array

// Load directly
const restored = await LinearModel.load(bytes)

// Or use the universal loader (auto-dispatches by typeId)
const { load } = require('@wlearn/core')
const restored = await load(bytes)  // works for any registered model type
```

The universal `load()` reads the bundle header, finds the registered loader for that model type, and returns a fitted estimator. This means you can load any wlearn model without knowing its type in advance -- useful for pipelines, ensemble systems, and model serving.

### Pipeline

Compose multiple steps into a single estimator. Steps are `[name, estimator]` tuples.

```js
const { Pipeline, load } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')

const model = await LinearModel.create({ task: 'classification' })
const pipe = new Pipeline([['clf', model]])

pipe.fit(X, y)
const preds = pipe.predict(X)

// Save/load works the same as individual models
const bytes = pipe.save()
const restored = await load(bytes)
restored.predict(X)

pipe.dispose()
restored.dispose()
```

### Parameters

```js
const params = model.getParams()     // { solver: 'L2R_LR', C: 1.0, ... }
model.setParams({ C: 10.0 })        // update before next fit()

// For AutoML: each model defines its search space
const space = LinearModel.defaultSearchSpace()
// { solver: { type: 'categorical', values: [...] }, C: { type: 'log_uniform', ... }, ... }
```

### Dispose

```js
model.dispose()  // frees WASM memory -- required
```

Models allocate memory on the WebAssembly linear heap. The JS garbage collector does not track this memory. Failing to call `dispose()` leaks memory. A `FinalizationRegistry` safety net will warn you in development, but do not rely on it.

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

Gradient-boosted trees for classification, regression, and ranking. Includes random forest mode.

```js
const { XGBModel } = require('@wlearn/xgboost')

// Binary classification
const clf = await XGBModel.create({
  objective: 'binary:logistic',
  max_depth: 6,
  eta: 0.3,
  nRounds: 100
})
clf.fit(X, y)
clf.predict(X)        // class labels (0 or 1)
clf.predictProba(X)   // probabilities, shape: rows * 2

// Multiclass
const mc = await XGBModel.create({
  objective: 'multi:softprob',
  num_class: 3,
  nRounds: 50
})

// Regression
const reg = await XGBModel.create({
  objective: 'reg:squarederror',
  nRounds: 100
})
reg.fit(X, y)
reg.predict(X)
reg.score(X, y)  // R-squared

// Random forest mode
const rf = await XGBModel.create({
  objective: 'binary:logistic',
  nRounds: 100,
  num_parallel_tree: 10,
  subsample: 0.8,
  colsample_bynode: 0.8
})
```

**Objectives:** `binary:logistic`, `multi:softprob`, `multi:softmax`, `reg:squarederror`, `reg:logistic`, `rank:pairwise`, and others.

**Key parameters:** `max_depth`, `eta` (learning rate), `nRounds` (number of boosting rounds), `subsample`, `colsample_bytree`, `lambda` (L2 reg), `alpha` (L1 reg), `num_parallel_tree` (for RF mode)

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

Pretrained Mitra Tab2D models for tabular data. ONNX-based inference via ONNX Runtime.

```js
const { MitraClassifier, MitraRegressor } = require('@wlearn/mitra')

// Classification
const clf = await MitraClassifier.create({ nFeatures: 10 })
clf.fit(X, y)
const preds = await clf.predict(Xtest)  // async (ONNX inference)

// Regression
const reg = await MitraRegressor.create({ nFeatures: 10 })
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

`StackingEnsemble` trains base models with out-of-fold predictions and feeds them to a meta-learner. `VotingEnsemble` averages predictions (soft vote) or takes majority class (hard vote). `BaggedEstimator` trains multiple copies of a single model on bootstrap samples.

### @wlearn/automl

Automated model selection: searches hyperparameter spaces across multiple model families, selects the best via cross-validation, and optionally builds an ensemble.

```js
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')
const { XGBModel } = require('@wlearn/xgboost')

const models = [
  ['linear', LinearModel, { task: 'classification' }],
  ['xgb', XGBModel, { task: 'classification' }]
]

const result = await autoFit(models, X, y, {
  strategy: 'random',    // 'random' | 'halving' | 'portfolio' | 'progressive'
  ensemble: true,         // build Caruana ensemble from top candidates
  ensembleSize: 20,
  refit: true,            // refit best model on full data
  onProgress: ({ phase, progress }) => console.log(phase, progress)
})

result.model           // best fitted estimator (or ensemble)
result.leaderboard     // ranked candidate results
result.bestScore       // best CV score
result.bestModelName   // e.g. 'xgb'
result.bestParams      // winning hyperparameters
```

`@wlearn/automl` requires at least one model package (e.g. `@wlearn/xgboost`) to do anything useful.

## Python

The Python `wlearn` package reads and writes the same WLRN bundles as the JS packages. Models trained in JS can be loaded in Python and vice versa.

```python
import wlearn.xgboost  # registers loader

# Load a bundle (produced by JS or Python)
model = wlearn.load(open('model.wlrn', 'rb').read())
preds = model.predict(X)
model.score(X, y)

# Save back to WLRN (loadable from JS)
bundle = model.save()
```

Python wrappers exist for: xgboost, liblinear, libsvm, nanoflann, lightgbm, ebm, stochtree, tsetlin, nn. Classical ML wrappers use native upstream packages. Neural models (nn) use polygrad via ctypes.

```
pip install wlearn               # bundle/registry/pipeline only
pip install wlearn[xgboost]      # + xgboost support
pip install wlearn[liblinear]    # + liblinear support
pip install wlearn[libsvm]       # + libsvm support
pip install wlearn[nanoflann]    # + k-nearest neighbors support
pip install wlearn[all]          # everything
```

Requires Python 3.9+.

## Cross-language interop

Bundles are portable between JS and Python. The WLRN format guarantees:

- **Identical blob bytes**: upstream serialization produces the same bytes regardless of host language
- **Identical predictions**: models loaded from the same bundle produce identical predictions in both languages (within floating-point tolerance)
- **Round-trip safe**: JS -> Python -> JS preserves model bytes exactly

Golden fixture tests verify all three directions:
- `fixtures/verify.mjs` validates JS-produced bundles
- `py/tests/test_compat.py` loads JS fixtures in Python, verifies format and predictions
- `fixtures/verify-py-bundles.mjs` validates Python-produced bundles back in JS

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

**Batch predictions are fast.** The predict loop runs entirely in C/WASM. One JS-to-WASM call predicts all rows -- no per-row overhead.

**Dispose promptly in loops.** If you are training many models (grid search, cross-validation), dispose each one before creating the next. WASM heap memory is not garbage collected.

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
npm install @wlearn/core         # just the core (bundle format, registry, pipeline)
```

Install everything at once (Node.js scripting):

```
npm install @wlearn/sdk
```

`@wlearn/sdk` re-exports all model classes, `autoFit`, `Pipeline`, `load`, metrics, and cross-validation utilities. It does not include `@wlearn/mitra` (requires ONNX Runtime peer dep); install that separately if needed. Browser users should import individual packages to avoid bundling unused WASM binaries.

Or install packages individually:

```
npm install @wlearn/core @wlearn/automl @wlearn/ensemble @wlearn/liblinear @wlearn/libsvm @wlearn/xgboost @wlearn/lightgbm @wlearn/nanoflann @wlearn/ebm @wlearn/xlearn @wlearn/stochtree @wlearn/tsetlin @wlearn/mitra onnxruntime-node
```

All packages are CommonJS (`"type": "commonjs"`). They work in Node.js 18+ and modern browsers.

```js
const { LinearModel } = require('@wlearn/liblinear')
```

## Repository structure

```
wlearn-org/wlearn             core monorepo (@wlearn/types, @wlearn/core, Python wlearn)
wlearn-org/liblinear-wasm     @wlearn/liblinear
wlearn-org/libsvm-wasm        @wlearn/libsvm
wlearn-org/xgboost-wasm       @wlearn/xgboost
wlearn-org/lightgbm-wasm      @wlearn/lightgbm
wlearn-org/nanoflann-wasm     @wlearn/nanoflann
wlearn-org/ebm-wasm           @wlearn/ebm
wlearn-org/xlearn-wasm        @wlearn/xlearn
wlearn-org/stochtree-wasm     @wlearn/stochtree
wlearn-org/tsetlin-wasm       @wlearn/tsetlin
wlearn-org/mitra-onnx         @wlearn/mitra
```

Model port repos carry WASM binaries and upstream C/C++ source as git submodules. They depend on `@wlearn/types` and `@wlearn/core` from the core repo. All Python lives in the core repo.

## License

MIT. Model packages carry their upstream licenses: BSD for LIBLINEAR, LIBSVM, and nanoflann; Apache-2.0 for XGBoost and xLearn; MIT for LightGBM, InterpretML, StochTree, TMU, and Mitra.
