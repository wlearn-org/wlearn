# Changelog

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
