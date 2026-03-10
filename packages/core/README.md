# @wlearn/core

Runtime core for wlearn: matrix helpers, bundle format, model registry, pipeline, preprocessing, metrics, and cross-validation. No WASM. No heavy dependencies.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/core
```

## Quick start

```js
const { Pipeline, load, accuracy, crossValScore } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')

// Build a pipeline
const model = await LinearModel.create({ task: 'classification' })
const pipe = new Pipeline([['clf', model]])

pipe.fit(X, y)
const preds = pipe.predict(X_test)
console.log('accuracy:', accuracy(y_test, preds))

// Save / load
const bytes = pipe.save()
const restored = await load(bytes)

pipe.dispose()
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

### Registry

Global loader dispatcher. Model packages register themselves on import.

- `register(typeId, loaderFn)` -- register a deserializer
- `load(bytes)` -- async: decode bundle, dispatch to registered loader
- `loadSync(bytes)` -- sync variant (limited to sync loaders)
- `getRegistry()` -- inspect registered loaders

### Pipeline

Sequential composition of transformers and estimators.

- `new Pipeline(steps)` -- `steps` is `[name, estimator][]`
- `pipe.fit(X, y)` -- fit all steps in order
- `pipe.predict(X)` -- transform + predict
- `pipe.score(X, y)` -- transform + score
- `pipe.save()` / `Pipeline.load(bytes)` -- serialize/deserialize
- `pipe.dispose()` -- dispose all steps

### Preprocessing

- `StandardScaler` -- zero mean, unit variance
- `MinMaxScaler` -- scale to [0, 1]
- `Preprocessor` -- base transformer class

### Metrics

Classification: `accuracy`, `confusionMatrix`, `precisionScore`, `recallScore`, `f1Score`, `logLoss`, `rocAuc`

Regression: `r2Score`, `meanSquaredError`, `meanAbsoluteError`

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
- `model.save()` / `Model.load(bytes)` -- serialize/deserialize.
- `model.dispose()` -- free resources.
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
