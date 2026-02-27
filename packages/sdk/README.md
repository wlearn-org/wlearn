# @wlearn/sdk

Convenience barrel that re-exports all wlearn model classes, AutoML, ensemble methods, pipeline, and core utilities in a single import.

Part of the [wlearn](https://github.com/wlearn-org/wlearn) project.

## Install

```bash
npm install @wlearn/sdk
```

## Usage

```js
import { XGBModel, LinearModel, autoFit, Pipeline, accuracy } from '@wlearn/sdk'

const model = await XGBModel.create({ task: 'classification' })
model.fit(X, y)
console.log('accuracy:', accuracy(y_test, model.predict(X_test)))
model.dispose()
```

## What is included

Models:

| Export | Package |
|--------|---------|
| `LinearModel` | `@wlearn/liblinear` |
| `SVMModel` | `@wlearn/libsvm` |
| `XGBModel` | `@wlearn/xgboost` |
| `LGBModel` | `@wlearn/lightgbm` |
| `KNNModel` | `@wlearn/nanoflann` |
| `EBMModel` | `@wlearn/ebm` |
| `TsetlinModel` | `@wlearn/tsetlin` |
| `BARTModel` | `@wlearn/stochtree` |
| `XLearnFMClassifier`, `XLearnFMRegressor`, `XLearnFFMClassifier`, `XLearnFFMRegressor`, `XLearnLRClassifier`, `XLearnLRRegressor` | `@wlearn/xlearn` |

AutoML and ensemble:

- `autoFit` from `@wlearn/automl`
- `VotingEnsemble`, `StackingEnsemble`, `BaggedEstimator` from `@wlearn/ensemble`

Core utilities:

- `Pipeline`, `load`, `loadSync`, `register`
- `encodeBundle`, `decodeBundle`, `validateBundle`
- `normalizeX`, `normalizeY`
- `StandardScaler`, `MinMaxScaler`, `Preprocessor`
- `accuracy`, `r2Score`, `f1Score`, `logLoss`, `rocAuc`, and other metrics
- `kFold`, `stratifiedKFold`, `trainTestSplit`, `crossValScore`

## Caveats

- **Node/scripting only.** Browser users should import individual packages so bundlers can tree-shake unused WASM binaries. The SDK pulls in all WASM modules (~30 MB total).
- Does not include `@wlearn/mitra` (requires `onnxruntime-node` or `onnxruntime-web` as a peer dependency).
- May lag behind individual model package releases.

## License

MIT
