# @wlearn/preprocess

Portable fitted tabular preprocessing for wlearn, implemented as a thin adapter
over Tranfi's generic prepared-transform API.

```js
const { Preprocessor, registerPreprocess } = require('@wlearn/preprocess')

const preprocessor = await Preprocessor.create({
  impute: 'auto',
  encode: 'auto',
  scale: 'standard'
})

const transformed = preprocessor.fitTransform([
  [1, 0],
  [2, 1],
  [NaN, 0]
])

const bundle = preprocessor.save()
await registerPreprocess()
const restored = await Preprocessor.load(bundle)
```

Construction and loading are asynchronous because the browser backend initializes
WASM. `fit`, `transform`, `fitTransform`, `save`, and `dispose` are synchronous.
Call `registerPreprocess()` explicitly before loading arbitrary or nested WLRN
bundles through `@wlearn/core`; a side-effect-only import is not a supported
registration route.

Tranfi owns generic analysis, immutable plans, and application. This package owns
the wlearn Transformer lifecycle, strict dense-matrix contract, WLRN persistence,
loader registration, and stable error mapping.

For compatibility, the package also re-exports core's existing `StandardScaler`
and `MinMaxScaler` constructors by identity. Their existing artifact type IDs and
loaders are unchanged; new mixed-type preprocessing should use `Preprocessor`.
