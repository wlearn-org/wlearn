# @wlearn/types

TypeScript interfaces and constants for wlearn. Zero runtime code. This is the contract that all wlearn packages implement against.

Part of the [wlearn](https://github.com/wlearn-org/wlearn) project.

## Install

```bash
npm install @wlearn/types
```

## Constants

```js
import { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE, DTYPE } from '@wlearn/types'

BUNDLE_MAGIC    // Uint8Array [0x57, 0x4c, 0x52, 0x4e] ('WLRN')
BUNDLE_VERSION  // 1
HEADER_SIZE     // 16 bytes
DTYPE           // { FLOAT32: 'float32', FLOAT64: 'float64', INT32: 'int32' }
```

## Types

Data types:

- `DenseMatrix` -- `{ data: Float32Array | Float64Array, rows, cols }`
- `CSRMatrix` -- `{ data, indices, indptr, rows, cols }` (compressed sparse row)
- `Matrix` -- `DenseMatrix | CSRMatrix`
- `Labels` -- `Int32Array | Float32Array | Float64Array`
- `TensorRef` -- zero-copy view descriptor for pipeline data routing

Estimator contract:

- `Estimator` -- `fit()`, `predict()`, `score()`, `save()`, `dispose()`, `getParams()`, `setParams()`
- `Classifier` -- extends Estimator with `predictProba()` and `classes`
- `Transformer` -- `fit()`, `transform()`, `fitTransform()`, `save()`, `dispose()`
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

MIT
