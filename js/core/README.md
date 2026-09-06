# @wlearn/core

Runtime core for wlearn: matrix helpers, bundle format, model registry, pipeline, preprocessing, metrics, and cross-validation. No WASM. No heavy dependencies.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/core @wlearn/liblinear
```

`@wlearn/core` contains no model backend. `@wlearn/liblinear` is included above
because the runnable example uses it.

## Quick start

```js
const { readFileSync, writeFileSync } = require('fs')
const { Pipeline, load, accuracy } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')

async function main() {
  const X = [[-2, -2], [-1, -1], [1, 1], [2, 2]]
  const y = new Int32Array([0, 0, 1, 1])
  const XTest = [[-1.5, -1.5], [1.5, 1.5]]
  const yTest = new Int32Array([0, 1])

  const model = await LinearModel.create({ task: 'classification' })
  const pipe = new Pipeline([['clf', model]])
  pipe.fit(X, y)

  const preds = pipe.predict(XTest)
  console.log('accuracy:', accuracy(yTest, preds)) // 1

  writeFileSync('pipeline.wlrn', pipe.save())
  // Importing @wlearn/liblinear above registered its bundle loaders.
  const restored = await load(readFileSync('pipeline.wlrn'))
  console.log(Array.from(restored.predict(XTest))) // [0, 1]
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
```

## API

### Matrix utilities

Convert user input to typed arrays for WASM consumption.

- `normalizeX(X)` -- `number[][] | DenseMatrix` to contiguous `DenseMatrix`
- `normalizeY(y)` -- preserves `Int32Array`, `Float32Array`, and `Float64Array`; converts `number[]` to `Float64Array`
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

Global loader dispatcher. Model packages register themselves on import. A fresh
process must import the matching package before calling generic `load()`; core
does not eagerly import optional backends.

- `register(typeId, loaderFn, { acceptsContext?, sync? })` -- register a deserializer; both flags default to `false`
- `load(bytes)` -- async: decode bundle, dispatch to registered loader
- `loadSync(bytes)` -- sync variant, only for loaders explicitly registered with `sync: true`
- `getRegistry()` -- inspect registered loaders
- `assertRequiredLoaders(manifest)` -- preflight all declared nested loaders

`load(bytes, { loaderOptions })` snapshots plain nested objects and arrays before
the first asynchronous loader runs. Every recursive child receives the same
read-only snapshot, so caller mutation during an `await` cannot change load
policy. Runtime identity objects such as typed cancellation flags are preserved.

### Pipeline

Sequential composition of transformers and estimators.

- `new Pipeline(steps)` -- `steps` is `[name, estimator][]`
- `pipe.fit(X, y)` -- fit all steps in order
- `pipe.predict(X)` -- transform + predict
- `pipe.classes` -- fitted class order forwarded by the final classifier
- `pipe.capabilities` -- final-estimator capability descriptor
- `pipe.score(X, y)` -- transform + score
- `pipe.save()` / `Pipeline.load(bytes)` -- serialize/deserialize WLRN bytes
- `pipe.dispose()` -- deterministic cleanup for long-running loops

`pipe.setParams({ stepName: params })` invalidates the fitted pipeline before
mutating the first selected child. Call `fit()` again before inference, including
when a child parameter setter throws partway through the update.
Unknown step names are rejected, and every selected child is checked for a
callable parameter setter before any fitted state or child configuration changes.

Base estimators still fit synchronously. If a Pipeline contains an orchestration
composite with asynchronous fit (for example a voting or stacking ensemble),
`pipe.fit()` Promise-lifts that operation and becomes fitted only after it resolves;
use `await pipe.fit(X, y)` for code that accepts either kind of child.
Concurrent `fit()`, `setParams()`, and `dispose()` calls are rejected while an
asynchronous Pipeline fit is pending, preventing late completion from resurrecting
or leaking a disposed composite.

The package publishes `index.d.ts`; `npm run test:types` checks the public
TypeScript surface against representative registry, bundle, scaler, and pipeline
usage.

### Preprocessing

- `StandardScaler` -- zero mean and population variance (`ddof=0`)
- `MinMaxScaler` -- scale the fitted range to [0, 1]

Core 0.3 removes the legacy state-only `Preprocessor`. Import it from
`@wlearn/preprocess` and replace `new Preprocessor(config)` with
`await Preprocessor.create(config)`. The adapter wraps Tranfi prepared transforms
and implements WLRN `save()`/load registration. Refit legacy `getState()` data
from its original training inputs; that state was not a portable WLRN artifact.

New scaler artifacts use the corrected `standard_scaler@2` and
`minmax_scaler@2` contracts. Both runtimes retain `@1` loaders so existing
constant-column artifacts preserve their historical inference behavior.

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

Apache-2.0
