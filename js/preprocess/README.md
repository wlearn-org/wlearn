# @wlearn/preprocess

Portable fitted tabular preprocessing for wlearn, implemented as a thin adapter
over Tranfi's generic prepared-transform API.

## Install

```bash
npm install @wlearn/preprocess
```

No Python or compiler is required. The Node entry uses Tranfi's native addon
when its prepared-transform API is available, otherwise the packaged WASM
backend. Browser bundlers select WASM; `@wlearn/preprocess/wasm` explicitly
selects it in Node too. `preprocessor.backend` reports the selected backend.

Existing lockfiles need an update to `@wlearn/preprocess` 0.1.1 and Tranfi 0.2.2
or newer; a fresh install of AutoML/SDK 0.3.0 already accepts these patches.
Native cancellation uses `cancelFlag`; WASM uses `cancelToken`. Backend choice
does not change the portable saved-plan format.

## Quick start

```js
const { Preprocessor } = require('@wlearn/preprocess')

async function main() {
  const preprocessor = await Preprocessor.create({
    impute: 'auto',
    encode: 'auto',
    scale: false
  })

  // Integer column x0 is inferred as categorical; decimal x1 is numeric.
  const transformed = preprocessor.fitTransform([
    [1, 10.5],
    [2, 20.5],
    [1, NaN],
    [2, 40.5]
  ])
  console.log(transformed.rows, transformed.cols) // 4 3

  const bundle = preprocessor.save()

  const restored = await Preprocessor.load(bundle)
  const next = restored.transform([[3, 50.5]])
  console.log(Array.from(next.data)) // [0, 0, 50.5]
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
```

Construction and loading are asynchronous because the browser backend initializes
WASM. `fit`, `transform`, `fitTransform`, `save`, and `dispose` are synchronous.
The transform result is a row-major dense matrix:
`{ dtype: 'float64', rows, cols, data: Float64Array }`.

## If you know scikit-learn

`fit()` learns imputation values, scaling statistics, categorical dictionaries,
and the output schema. `transform()` reuses that frozen state without learning.
`fitTransform()` performs both operations on the training data, equivalent to
analyze/finalize followed by a second application pass in Tranfi.

This V1 adapter accepts dense numeric input, not string columns or DataFrames.
A column is inferred as categorical only when every observed nonmissing value is
a finite integer and there are between 2 and `maxCategories` distinct values.
Missing values are ignored during inference. All other columns, including
constant columns, are numeric. Inspect `inputSchema` and `outputSchema` after fit
when column expansion matters.

## Options

| Option | Values | Default and behavior |
|--------|--------|----------------------|
| `impute` | `'auto'`, `'mean'`, `'median'`, `'zero'`, `false` | `'auto'`: numeric mean and categorical mode |
| `encode` | `'auto'`, `'onehot'`, `'label'`, `false` | `'auto'` resolves to one-hot |
| `scale` | `'standard'`, `'minmax'`, `false` | `false`; scaling applies to numeric value columns |
| `maxCategories` | integer >= 2 | `20`; maximum distinct integers for a column to be inferred as categorical; exceeding it resolves numeric |
| `unknownCategory` | policy depends on encoding | one-hot: `'all_zero'`; label: `'sentinel'`; either can use `'error'` |
| `allMissing` | `'zero'` or `'error'` | `'zero'` when imputation is enabled |
| `maxOutputColumns` | positive integer | `65536` |
| `maxOutputElements` | positive integer | `100000000` per transform call |

Unknown option names and incompatible policy combinations are rejected. Resource
limits are part of the public contract so an unexpected category expansion cannot
silently allocate an unbounded output.

## Pipeline use

```js
const { Pipeline } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')
const { Preprocessor } = require('@wlearn/preprocess')

async function makePipeline() {
  const prep = await Preprocessor.create({ scale: 'standard' })
  const model = await LinearModel.create({ task: 'classification' })
  return new Pipeline([['prep', prep], ['model', model]])
}
```

Pipeline persistence nests the preprocessor's WLRN bundle. Before loading such a
Pipeline in a fresh process, import its model packages and call
`await registerPreprocess()` so every declared loader is available.

## JavaScript API

- `await Preprocessor.create(config?, runtimeOptions?)` -- initialize the backend.
- `fit(X, y?)` -- learn a plan; `y` is accepted for Pipeline compatibility and ignored.
- `transform(X)` -- apply the fitted plan without updating it.
- `fitTransform(X, y?)` -- fit, then transform the same rows.
- `getParams()` / `setParams(patch)` -- inspect or update resolved configuration; an update clears fitted state.
- `save()` / `await Preprocessor.load(bytes)` -- WLRN persistence.
- `inputSchema` / `outputSchema` -- fitted schema metadata.
- `dispose()` -- release prepared-plan resources in long-running processes.

Direct `Preprocessor.load()` initializes its own backend. In a fresh process,
`registerPreprocess()` is required before loading arbitrary or nested WLRN
bundles through `@wlearn/core`; a side-effect-only import is not a supported
registration route.

## Python counterpart

```bash
pip install 'wlearn[preprocess]'
```

```python
import numpy as np
from wlearn.preprocess import Preprocessor

prep = Preprocessor(impute='auto', encode='auto', scale='standard')
X_out = prep.fit_transform(np.array([
    [1.0, 10.5],
    [2.0, 20.5],
    [1.0, np.nan],
    [2.0, 40.5],
]))
prep.save('preprocessor.wlrn')
restored = Preprocessor.load('preprocessor.wlrn')
```

Python construction and loading are synchronous and transformed output is a 2-D
NumPy `float64` array. Python uses snake_case methods and keyword names, while the
saved WLRN artifact has the same cross-language contract.

## Ownership boundary

Tranfi owns generic streaming analysis, immutable prepared plans, and application.
This package owns the wlearn Transformer lifecycle, strict dense-matrix contract,
WLRN persistence, loader registration, and stable error mapping. Ordinary wlearn
pipelines should use this adapter; use Tranfi directly when building a general
byte-stream or typed-batch data-processing integration.

For compatibility, the package also re-exports core's existing `StandardScaler`
and `MinMaxScaler` constructors by identity. Corrected new fits write
`standard_scaler@2` and `minmax_scaler@2`; core retains the `@1` loaders and legacy
constant-column inference for existing artifacts. New inferred numeric/categorical
preprocessing over dense numeric matrices should use the `Preprocessor` defined by
this package.

## License

Apache-2.0

Per-column overrides use canonical input IDs (`x0`, `x1`, …):

```js
const pre = await Preprocessor.create({
  scale: 'standard',
  columns: {
    x0: { kind: 'numeric', scale: false },
    x1: { categories: [0, 2, 5] },
    x2: { kind: 'categorical', encode: 'label' }
  }
})
```

Each column inherits global options except those explicitly overridden. Supported
column options are `kind`, `categories`, `impute`, `encode`, `scale`, `maxCategories`,
`allMissing`, and `unknownCategory`. `kind` is `numeric`, `categorical`, or `infer`.
A `categories` array implies `categorical`; it must contain unique finite numbers
and is sorted numerically. The adapter uses float64 category tags. Encoding keeps
all declared categories, including those absent during fitting, so output width and
ordinals remain stable across refits. Unknown training values follow the encoder
policy and do not compete for the mode; a zero fallback must be in the dictionary.
Fixed categories require encoding or mode imputation. String categories are not
supported.

`setParams({ columns: … })` replaces the override map; an empty map removes it.
Global updates affect inherited options and preserve explicit column overrides.
Any successful parameter update clears fitted state. Column IDs outside the fitted
input width fail validation. Omitted or empty overrides preserve existing resolved
configuration and bundle bytes. Python accepts the same mapping through
`Preprocessor(columns={...})` or `Preprocessor(config)`.
