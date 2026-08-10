# @wlearn/core

Runtime core for wlearn: matrix helpers, bundle format, model registry, pipeline, preprocessing, metrics, and cross-validation. No WASM. No heavy dependencies.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/core
```

## Quick start

```js
const { readFileSync, writeFileSync } = require('fs')
const { Pipeline, load, accuracy, crossValScore } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')

// Build a pipeline
const model = await LinearModel.create({ task: 'classification' })
const pipe = new Pipeline([['clf', model]])

pipe.fit(X, y)
const preds = pipe.predict(X_test)
console.log('accuracy:', accuracy(y_test, preds))

// Save / load
writeFileSync('pipeline.wlrn', pipe.save())
const restored = await load(readFileSync('pipeline.wlrn'))
```

## API

### Matrix utilities

Convert user input to typed arrays for WASM consumption.

- `normalizeX(X)` -- `number[][] | DenseMatrix` to contiguous `DenseMatrix`
- `normalizeY(y)` -- `number[] | TypedArray` to `Float64Array`
- `makeDense(data, rows, cols)` -- create `DenseMatrix` from typed array
- `validateMatrix(m)` -- validate matrix structure and dimensions

### Bundle format

Portable binary format for model artifacts. Language-agnostic, deterministic.

- `encodeBundle(manifest, artifacts)` -- encode to `Uint8Array`
- `decodeBundle(bytes)` -- decode to `{ manifest, toc, blobs }`
- `validateBundle(bytes)` -- decode + verify SHA-256 hashes

Artifacts are `{ id, mediaType?, data: Uint8Array }` objects. The manifest includes `typeId`, `params`, and `metadata`. See `@wlearn/types` for full shapes.

Canonical v1 writers always emit `bundleVersion`, `requires`, `params`, and an
`artifacts` declaration matching the TOC. Each TOC record has exactly `id`,
`offset`, `length`, `sha256`, and `mediaType`; each artifact declaration has the
same fields except `offset`. Nested WLRN artifacts must themselves be canonical.
Use `validateBundle(bytes, { allowLegacyManifest: false })` to prove canonical
conformance.

The default reader is a compatibility-ingestion path for historical v1 bundles.
It additionally permits missing `requires`, `params`, `artifacts`, and TOC
`mediaType`, non-canonical TOC ordering, and extensions on fixed TOC/declaration
records. It still enforces the v1 header and typeId, bounded UTF-8/portable JSON,
exact blob coverage, hashes, recursion limits, and every field that is present.
Because historical bundles may omit `requires`, they cannot guarantee complete
nested-loader preflight. Writers never emit this legacy shape. Legacy reads remain
supported through the current major release; any removal requires a major release,
a migration tool, and advance deprecation notice.

### Registry

Global loader dispatcher. Model packages register themselves on import.

- `register(typeId, loaderFn)` -- register a deserializer
- `load(bytes)` -- async: decode bundle, dispatch to registered loader
- `loadSync(bytes)` -- sync variant (limited to sync loaders)
- `getRegistry()` -- inspect registered loaders
- `assertRequiredLoaders(manifest)` -- preflight all declared nested loaders

### Pipeline

Sequential composition of transformers and estimators.

- `new Pipeline(steps)` -- `steps` is `[name, estimator][]`
- `pipe.fit(X, y)` -- fit all steps in order
- `pipe.predict(X)` -- transform + predict
- `pipe.score(X, y)` -- transform + score
- `pipe.save()` / `Pipeline.load(bytes)` -- serialize/deserialize WLRN bytes
- `pipe.dispose()` -- deterministic cleanup for long-running loops

### Preprocessing

- `StandardScaler` -- zero mean, unit variance
- `MinMaxScaler` -- scale to [0, 1]
- `Preprocessor` -- base transformer class

### Metrics

Classification: `accuracy`, `confusionMatrix`, `precisionScore`, `recallScore`, `f1Score`, `logLoss`, `rocAuc`

Regression: `r2Score`, `meanSquaredError`, `meanAbsoluteError`

Metrics accept `sampleWeight` / `sample_weight` where defined. Classification metrics support `binary`, `micro`, `macro`, and `weighted` averaging. `rocAuc` supports binary scores plus multiclass `multiClass: 'ovr' | 'ovo'`; undefined metrics can throw, warn, or return `NaN`.

```js
const { accuracy, f1Score, r2Score } = require('@wlearn/core')

accuracy(yTrue, yPred)                        // number
f1Score(yTrue, yPred, { average: 'macro' })   // number
r2Score(yTrue, yPred)                         // number
```

### Cross-validation

- `kFold(n, k?, opts?)` -- k-fold split indices
- `stratifiedKFold(y, k?, opts?)` -- stratified k-fold
- `trainTestSplit(n, opts?)` -- single train/test split
- `crossValScore(ModelClass, X, y, opts?)` -- evaluate with CV
- `getScorer(name)` -- get scoring function by name (`'accuracy'`, `'r2'`, `'neg_mse'`)

### Ecosystem primitives

These are the stable objects for apps, AutoML, benchmarks, and agents. They are optional. If you only want to fit, predict, score, and save a model, use the estimator and pipeline APIs above.

- `createTask()` / `validateTask()` -- dataset, feature schema, labels, groups, row roles
- `createPrediction()` / `validatePrediction()` -- responses, probabilities, class order, truth
- `listMeasures()` / `evaluateMetricSet()` -- discover and score metrics, including weighted and multiclass measures
- `createResamplingPlan()` -- deterministic holdout, k-fold, stratified, group, time-series, and sliding row/index/period splits
- `Archive` -- candidate records, scores, timings, errors, and archive-level leaderboards

```js
const { createPrediction, createResamplingPlan, evaluateMetricSet, Archive } = require('@wlearn/core')

const pred = createPrediction({ truth: yTest, response, proba, classes })
const scores = evaluateMetricSet(['accuracy', 'log_loss', 'roc_auc_ovr'], pred)
const plan = createResamplingPlan({ strategy: 'sliding_period', index: dates, period: 'day', lookback: 14 })

const archive = new Archive({ measures: ['accuracy'], primaryMeasure: 'accuracy' })
archive.add({ trialId: 'xgb-0', candidateId: 'xgb-depth6', scores, status: 'ok' })
archive.leaderboard()
```

### createModelClass

Factory for building unified model classes from separate classifier/regressor implementations. Handles automatic task detection, async WASM pre-loading, and lifecycle management.

```js
const { createModelClass } = require('@wlearn/core')

// Task-agnostic model (same class handles both tasks)
const XGBModel = createModelClass(XGBModelImpl, XGBModelImpl, {
  name: 'XGBModel',
  load: loadXGB   // async WASM loader, called in create()
})

// Split model (separate classifier/regressor classes)
const MLPModel = createModelClass(MLPClassifier, MLPRegressor, {
  name: 'MLPModel'
})
```

The returned class supports:

- `Model.create(params)` -- async factory. Pass `task: 'classification'` or `task: 'regression'` to select explicitly, or omit to auto-detect from `y` at `fit()` time.
- `model.fit(X, y)` -- trains the model. Auto-detects task from labels if not set.
- `model.predict(X)`, `model.predictProba(X)`, `model.score(X, y)` -- proxied to inner model.
- `model.save()` / `Model.load(bytes)` -- serialize/deserialize WLRN bytes.
- `model.dispose()` -- deterministic cleanup for long-running loops.
- `model.task` -- the detected or specified task.
- Extra methods and getters from the inner classes are discovered and proxied automatically.

Auto-detection rules: if `y` is `Int32Array`, task is classification. Otherwise, if any value is non-integer, task is regression. If all values are integers and there are 20 or fewer unique values, task is classification; otherwise regression.

### Errors

`WlearnError`, `BundleError`, `RegistryError`, `ValidationError`, `NotFittedError`, `DisposedError`

### Utilities

- `sha256Sync(data)` -- pure JS SHA-256
- `makeLCG(seed?)` -- deterministic LCG random number generator
- `shuffle(arr, rng)` -- in-place shuffle
- `isPromiseLike(x)` / `lift(x, fn)` -- MaybePromise utilities

## License

MIT
