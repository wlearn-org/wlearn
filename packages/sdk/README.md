# @wlearn/sdk

Convenience barrel that re-exports all wlearn model classes, AutoML, ensemble methods, pipeline, and core utilities in a single import.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/sdk
```

## Usage

```js
const { XGBModel, LinearModel, autoFit, Pipeline, accuracy } = require('@wlearn/sdk')

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
| `XLearnLR`, `XLearnFM`, `XLearnFFM` | `@wlearn/xlearn` |
| `MLPModel`, `TabMModel`, `NAMModel` | `@wlearn/nn` |
| `RFModel`, `loadRF` | `@wlearn/rf` |
| `GAMModel`, `loadGAM` | `@wlearn/gam` |
| `ClusterModel`, `silhouette`, `calinskiHarabasz`, `daviesBouldin`, `adjustedRand`, `loadCluster` | `@wlearn/cluster` |
| `MitraModel`, `registerMitraLoaders` | `@wlearn/mitra` (optional) |

All model classes above are unified wrappers built with `createModelClass`. They accept an optional `task` parameter (`'classification'` or `'regression'`) and auto-detect the task from labels if omitted. Split classes (`XLearnFMClassifier`, `MLPClassifier`, etc.) are also re-exported for backward compatibility.

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
- `@wlearn/mitra` is an optional peer dependency (requires `onnxruntime-node` or `onnxruntime-web`). If installed, its exports are available; otherwise they are `undefined`.
- May lag behind individual model package releases.

## License

Apache-2.0
