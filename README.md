# wlearn

Classical machine learning that runs entirely in the browser and Node.js. No server, no Python runtime, no data leaving your machine.

wlearn compiles battle-tested C/C++ ML libraries (LIBLINEAR, LIBSVM, XGBoost, nanoflann, LightGBM) to WebAssembly and wraps them in a unified, sklearn-style JavaScript API. Train a model, serialize it to a portable binary bundle, load it anywhere -- same predictions, same format, JS or Python.

## Why

Most ML libraries require Python and a server. That means network round-trips, data privacy concerns, and infrastructure to manage. For many use cases -- on-device inference, privacy-sensitive data, offline apps, rapid prototyping -- you just want the model to run where the data already is.

WebAssembly makes this possible. The same optimized C code that powers scikit-learn's SVM and linear classifiers can compile to WASM and run at near-native speed in any modern browser or Node.js process. wlearn packages these compilers into npm modules with a clean JS API, so you can `npm install` a classifier the same way you `pip install` one.

## How it works

**WASM ports, not reimplementations.** Each model package compiles the original upstream C/C++ source (LIBLINEAR v2.50, LIBSVM v3.37, etc.) to WebAssembly via Emscripten. The numerical results match the native libraries.

**Unified API.** Every model follows the same pattern: async construction (WASM must load), then synchronous `fit`, `predict`, `score`, `save`, `dispose`. No surprises if you know scikit-learn.

**Portable bundles.** `save()` produces a self-describing binary bundle (format: WLRN v1) containing the model weights, hyperparameters, and a type identifier. `load()` reads the bundle and dispatches to the right loader automatically. Bundles are language-agnostic -- the Python `wlearn` package reads the same files.

**Mandatory resource management.** WASM models allocate linear memory that the JavaScript garbage collector cannot see. Every model has a `dispose()` method. Call it when you are done.

## Quick start

```
npm install @wlearn/liblinear
```

```js
import { LinearModel } from '@wlearn/liblinear'

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

### Model ports

| Package | Upstream | What it does |
|---------|----------|--------------|
| `@wlearn/liblinear` | LIBLINEAR v2.50 | Linear SVM and logistic regression. Fast on large sparse datasets. |
| `@wlearn/libsvm` | LIBSVM v3.37 | Kernel SVM (RBF, polynomial, sigmoid). Classification, regression, one-class novelty detection. |
| `@wlearn/xgboost` | XGBoost v2.1.4 | Gradient-boosted trees, random forests. Classification, regression, ranking. |
| `@wlearn/lightgbm` | LightGBM | Gradient-boosted trees, fast histogram-based. (Planned) |
| `@wlearn/nanoflann` | nanoflann v1.6.3 | k-nearest neighbors via KD-trees. Classification and regression. |
| `@wlearn/ebm` | InterpretML | Explainable boosting machines. (Planned) |

## API overview

Every model package exports a model class that implements the same contract.

### Construction

WASM modules load asynchronously. Use the static `create()` factory:

```js
const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
```

After construction, all methods are synchronous.

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
import { load } from '@wlearn/core'
const restored = await load(bytes)  // works for any registered model type
```

The universal `load()` reads the bundle header, finds the registered loader for that model type, and returns a fitted estimator. This means you can load any wlearn model without knowing its type in advance -- useful for pipelines, ensemble systems, and model serving.

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
import { LinearModel, Solver } from '@wlearn/liblinear'

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
import { SVMModel, SVMType, Kernel } from '@wlearn/libsvm'

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
import { XGBModel } from '@wlearn/xgboost'

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

### @wlearn/nanoflann

k-nearest neighbors via KD-trees. Fast exact neighbor search for classification and regression.

```js
import { KNNModel } from '@wlearn/nanoflann'

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

**Notes:**
- L2 distances are Euclidean (sqrt of squared distance). L1 distances are Manhattan.
- k is clamped to n_samples when k > n_samples.
- Classification ties are broken by smallest class label.

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

Python wrappers are thin: `wlearn.xgboost` wraps the native `xgboost` package, `wlearn.liblinear` wraps `liblinear-official`, `wlearn.libsvm` wraps `libsvm-official`, `wlearn.nanoflann` wraps `pynanoflann`. Model blob bytes are identical across languages (same upstream serialization format), so bundles are fully interoperable.

## Cross-language interop

Bundles are portable between JS and Python. The WLRN format guarantees:

- **Identical blob bytes**: upstream serialization (xgboost UBJ, liblinear text, libsvm text) produces the same bytes regardless of host language
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

The `typeId` field (e.g., `wlearn.liblinear.classifier@1`, `wlearn.libsvm.regressor@1`) tells the loader registry which deserializer to use. This makes bundles portable across languages and runtimes.

```js
import { decodeBundle } from '@wlearn/core'

const { manifest, toc, blobs } = decodeBundle(bundle)
console.log(manifest.typeId)    // 'wlearn.liblinear.classifier@1'
console.log(manifest.params)    // { solver: 'L2R_LR', C: 1.0, ... }
console.log(toc[0].id)          // 'model'
console.log(toc[0].sha256)      // hex hash of model blob
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

Set `coerce: 'warn'` to get notified when implicit copies happen:

```js
const model = await LinearModel.create({ solver: 'L2R_LR', coerce: 'warn' })
```

**Batch predictions are fast.** The predict loop runs entirely in C/WASM. One JS-to-WASM call predicts all rows -- no per-row overhead.

**Dispose promptly in loops.** If you are training many models (grid search, cross-validation), dispose each one before creating the next. WASM heap memory is not garbage collected.

## Install

### JavaScript

```
npm install @wlearn/liblinear    # linear SVM + logistic regression
npm install @wlearn/libsvm       # kernel SVM
npm install @wlearn/xgboost      # gradient-boosted trees + random forests
npm install @wlearn/nanoflann    # k-nearest neighbors (KD-tree)
npm install @wlearn/core         # just the core (bundle format, registry, pipeline)
```

All packages are ESM-only (`"type": "module"`). They work in Node.js 18+ and modern browsers.

### Python

```
pip install wlearn               # bundle/registry/pipeline only
pip install wlearn[xgboost]      # + xgboost support
pip install wlearn[liblinear]    # + liblinear support
pip install wlearn[libsvm]       # + libsvm support
pip install wlearn[nanoflann]    # + k-nearest neighbors support
pip install wlearn[all]          # everything
```

Requires Python 3.9+. Model wrappers are thin: they wrap native upstream packages (`xgboost`, `liblinear-official`, `libsvm-official`, `pynanoflann`) and add WLRN bundle save/load.

## Repository structure

```
wlearn-org/wlearn             core monorepo (@wlearn/types, @wlearn/core, Python wlearn)
wlearn-org/liblinear-wasm     @wlearn/liblinear (separate repo)
wlearn-org/libsvm-wasm        @wlearn/libsvm (separate repo)
wlearn-org/xgboost-wasm       @wlearn/xgboost (separate repo)
wlearn-org/nanoflann-wasm     @wlearn/nanoflann (separate repo)
```

Model port repos carry large WASM binaries and upstream C/C++ source as git submodules. They depend on `@wlearn/types` and `@wlearn/core` from this repo. All Python lives in the core repo.

## License

MIT. Model packages carry their upstream licenses: BSD for LIBLINEAR, LIBSVM, and nanoflann, Apache-2.0 for XGBoost, MIT for LightGBM.
