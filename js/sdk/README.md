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

This simple path is still the default. The task/prediction/archive helpers are exported for apps, AutoML provenance, and agents; they are not required for ordinary model usage.

## API

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
| `BasisClassifier`, `BasisRegressor`, `BasisTransformer`, `loadBasis` | `@wlearn/basis` |
| `SymbolicRegressor`, `SymbolicClassifier`, `FormulaTransformer` | `@wlearn/sym` |
| Calibration, conformal intervals/sets and risk controllers | `@wlearn/uncertainty` |
| `GAMModel`, `loadGAM` | `@wlearn/gam` |
| `ClusterModel`, `silhouette`, `calinskiHarabasz`, `daviesBouldin`, `adjustedRand`, `loadCluster` | `@wlearn/cluster` |
| `BayesianSearch`, `BayesianStrategy` | `@wlearn/automl` |
| `BayesianOptimizer`, `compileSpace`, `encodeParams`, `decodeParams` | `@wlearn/bo` |
| `MitraClassifier`, `MitraRegressor`, `registerMitraLoaders` | `@wlearn/mitra` (optional; requires an ONNX source) |

Unified classes accept an optional `task` parameter (`'classification'` or `'regression'`) and auto-detect the task from labels if omitted. Split classes (`XLearnFMClassifier`, `MLPClassifier`, etc.) are exported for explicit task-specific imports.

AutoML and ensemble:

- `autoFit` from `@wlearn/automl`
- Bayesian search from `@wlearn/automl`, powered by `@wlearn/bo`
- `VotingEnsemble`, `StackingEnsemble`, `BaggedEstimator` from `@wlearn/ensemble`

Core utilities:

- `Pipeline`, `load`, `loadSync`, `register`
- `encodeBundle`, `decodeBundle`, `validateBundle`
- `normalizeX`, `normalizeY`
- `StandardScaler`, `MinMaxScaler`
- `accuracy`, `r2Score`, `f1Score`, `logLoss`, `rocAuc`, and other metrics
- `kFold`, `stratifiedKFold`, `trainTestSplit`, `crossValScore`
- `createTask`, `createPrediction`, `evaluateMetricSet`, `createResamplingPlan`, `Archive`

The SDK is a barrel only. Model behavior, persistence, and memory ownership stay in the underlying packages.

`Preprocessor` is re-exported from `@wlearn/preprocess`. In SDK 0.3, replace
`new Preprocessor(config)` with `await Preprocessor.create(config)`. Fit and
transform remain synchronous after construction; fitted preprocessing saves in
WLRN Pipelines. Legacy core `getState()` objects require refitting from training
data; they are not adapter artifacts.

## Caveats

- **Node/scripting only.** Browser users should import individual packages so bundlers can tree-shake unused WASM binaries. The SDK pulls in its complete model dependency set.
- `@wlearn/mitra` is an optional peer dependency (requires `onnxruntime-node` or `onnxruntime-web`). If installed, its exports are available; otherwise they are `undefined`.
- The SDK is released only after its exact model/core dependency set is aligned; it may lag behind individual package releases.

## License

Apache-2.0
