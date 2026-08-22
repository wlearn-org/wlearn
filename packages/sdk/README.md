# @wlearn/sdk

Convenience barrel that re-exports all wlearn model classes, AutoML, ensemble methods, pipeline, and core utilities in a single import.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/sdk
```

## Usage

```js
const { LinearModel, accuracy } = require('@wlearn/sdk')

async function main() {
  const X = [[-2, -2], [-1, -1], [1, 1], [2, 2]]
  const y = new Int32Array([0, 0, 1, 1])
  const XTest = [[-1.5, -1.5], [1.5, 1.5]]
  const yTest = new Int32Array([0, 1])

  const model = await LinearModel.create({ task: 'classification' })
  model.fit(X, y)
  console.log('accuracy:', accuracy(yTest, model.predict(XTest))) // 1
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
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

- **Node/scripting only.** Browser users should import individual packages so bundlers can tree-shake unused WASM binaries. The SDK pulls in its complete model dependency set.
- `@wlearn/mitra` is an optional peer dependency (requires `onnxruntime-node` or `onnxruntime-web`). If installed, its exports are available; otherwise they are `undefined`.
- The SDK is released only after its exact model/core dependency set is aligned; it may lag behind individual package releases.

## License

Apache-2.0
