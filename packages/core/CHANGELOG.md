# Changelog

## 0.2.0

- Add `createModelClass(ClassifierCls, RegressorCls, opts)` factory for unified model wrappers
- Automatic task detection from labels (classification vs regression)
- `sameClass` optimization: when both classes are identical, inner model is created in `create()` without requiring `task`
- `opts.load` option for async WASM pre-loading so `fit()` stays synchronous
- `opts.name` option for error messages
- Disposed state tracking with distinct error messages for disposed vs not-fitted
- Auto-discovery of extra methods/getters from inner class prototypes
- Proxy for static `defaultSearchSpace()` and `budgetSpec()` methods
- Add `detectTask(y)` utility function

## 0.1.0

- Initial release
- Matrix helpers: normalizeX, normalizeY, makeDense, validateMatrix
- Bundle format: encode/decode/validate WLRN v1 bundles
- Registry: global loader dispatcher with register/load/loadSync
- Pipeline: sequential estimator pipeline with save/load
- Preprocessing: Preprocessor (impute/encode/scale), StandardScaler, MinMaxScaler
- Metrics: accuracy, r2Score, MSE, MAE, confusion matrix, precision, recall, F1, log loss, ROC AUC
- Cross-validation: kFold, stratifiedKFold, trainTestSplit, crossValScore
- RNG: deterministic LCG PRNG and shuffle
- Lift: MaybePromise utilities for sync/async estimator interop
