# @wlearn/core

Runtime core for wlearn: matrix helpers, bundle format, model registry, pipeline, preprocessing, metrics, and cross-validation. No WASM. No heavy dependencies.

Part of the [wlearn](https://github.com/wlearn-org/wlearn) project.

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

### Errors

`WlearnError`, `BundleError`, `RegistryError`, `ValidationError`, `NotFittedError`, `DisposedError`

### Utilities

- `sha256Sync(data)` -- pure JS SHA-256
- `makeLCG(seed?)` -- deterministic LCG random number generator
- `shuffle(arr, rng)` -- in-place shuffle
- `isPromiseLike(x)` / `lift(x, fn)` -- MaybePromise utilities

## License

MIT
