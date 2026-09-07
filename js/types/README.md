# @wlearn/types

TypeScript interfaces plus a small CommonJS constants module for wlearn. This is
the shared contract that wlearn packages implement against; it has no runtime
dependencies or model logic.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/types
```

## API

### Constants

```js
const { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE, DTYPE } = require('@wlearn/types')

BUNDLE_MAGIC    // Uint8Array [0x57, 0x4c, 0x52, 0x4e] ('WLRN')
BUNDLE_VERSION  // 1
HEADER_SIZE     // 16 bytes
DTYPE           // { FLOAT32: 'float32', FLOAT64: 'float64', INT32: 'int32' }
```

Only constants are runtime values in this package. Import implementations such
as `Pipeline`, registry functions, metrics, and error classes from `@wlearn/core`.

### Types

Data types:

- `DenseMatrix` -- `{ data: Float32Array | Float64Array, rows, cols }`
- `CSRMatrix` -- `{ data, indices, indptr, rows, cols }` (compressed sparse row)
- `Matrix` -- `DenseMatrix | CSRMatrix`
- `Labels` -- `Int32Array | Float32Array | Float64Array`
- `TensorRef` -- descriptor for planned zero-copy routing; current Pipeline uses dense host matrices

Estimator contract:

- `Estimator` -- `fit()`, `predict()`, `score()`, `save()`, `getParams()`, `setParams()`, plus `dispose()` for deterministic cleanup
- `Classifier` -- extends Estimator with `predictProba()` and `classes`
- `Transformer` -- `fit()`, `transform()`, `fitTransform()`, `save()`, plus `dispose()` for deterministic cleanup
- `Capabilities` -- runtime feature flags (`classifier`, `predictProba`, `csr`, etc.)

AutoML:

- `SearchParam` -- hyperparameter distribution (`categorical`, `uniform`, `log_uniform`, `int_uniform`)
- `SearchSpace` -- `Record<string, SearchParam>` for model-provided search spaces
- `AutoFitOpts`, `AutoFitResult` -- options and result for `autoFit()`

Bundle format:

- `BundleManifest` -- manifest with `typeId`, `params`, `metadata`
- `BundleTOCEntry` -- `{ id, offset, length, sha256, mediaType }`
- `LoaderFn` -- `(manifest, toc, blobs) => Estimator | Promise<Estimator>`

## License

Apache-2.0
