# @wlearn/automl

Automated model selection for wlearn. Searches over model families and hyperparameters, runs cross-validation, and optionally builds a Caruana ensemble from top candidates.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/automl
```

Requires at least one model package (e.g. `@wlearn/xgboost`) to do anything useful.

## Quick start

```js
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')
const { XGBModel } = require('@wlearn/xgboost')

const models = [
  { name: 'linear', classId: 'wlearn.liblinear.classifier@1',
    portfolioKey: 'linear', cls: LinearModel, params: { task: 'classification' } },
  { name: 'xgb', classId: 'wlearn.xgboost.classifier@1',
    portfolioKey: 'xgb', cls: XGBModel, params: { task: 'classification' } }
]

const result = await autoFit(models, X, y, {
  scoring: 'accuracy',
  cv: 5,
  nIter: 20,
  ensemble: true,
  ensembleSize: 10,
  refit: true
})

result.model         // best fitted estimator (or ensemble)
result.leaderboard   // ranked candidate results
result.archive       // structured Archive of ok/failed trials
result.bestScore     // best CV score
result.bestModelName // e.g. 'xgb'
result.bestParams    // { model, preprocess }
result.bestCandidate // structured model + preprocessing identity
```

For the simple path, this is the whole API: use `result.model` for prediction and `result.leaderboard` for ranked candidates. `result.archive` is optional metadata for provenance, failed candidates, and agent/API workflows.

## API

- `autoFit(models, X, y, opts)` runs model search and returns `{ model, leaderboard, archive, bestScore, bestModelName, bestParams }`.
- `RandomSearch`, `SuccessiveHalvingSearch`, `PortfolioSearch`, `ProgressiveSearch`, `BayesianSearch` expose lower-level `.fit(X, y)` search classes.
- `RandomStrategy`, `HalvingStrategy`, `PortfolioStrategy`, `ProgressiveStrategy`, `BayesianStrategy` are strategy objects used by the executor layer.
- `Executor` evaluates candidates and folds.
- `Leaderboard` stores ranked candidate results.
- `PORTFOLIO` / `getPortfolio(task)` expose built-in model configurations.
- `sampleParam`, `sampleConfig`, `randomConfigs`, `gridConfigs` sample search-space IR.
- `registerBayesianSearch()` enables Bayesian search in environments that load strategy plugins explicitly.

## Search strategies

- `RandomSearch` -- random hyperparameter sampling (default)
- `SuccessiveHalvingSearch` -- early stopping with increasing resource allocation
- `PortfolioSearch` -- predefined portfolio of known-good configurations
- `ProgressiveSearch` -- progressive resource allocation
- `strategy: 'bayesian'` -- Bayesian optimization powered by `@wlearn/bo`, included as an AutoML dependency

Each model provides a default search space via `Model.defaultSearchSpace()`. AutoML samples from these automatically.

## Preprocessing

`@wlearn/automl` uses `@wlearn/preprocess` and Tranfi prepared transforms. A
fresh preprocessor is fitted inside every CV and OOF training fold, so validation
rows never influence imputation, scaling statistics, category dictionaries, or
output columns. The winning preprocessing plan is fitted again on the full data
only during refit.

```js
const result = await autoFit(models, X, y, {
  preprocess: {
    impute: 'auto',
    encode: 'onehot',
    scale: 'standard'
  }
})
```

Pass an explicit template list to compare several fixed preprocessing policies:

```js
preprocess: [
  {
    templateId: 'plain',
    typeId: 'wlearn.preprocess.tabular@1',
    params: { impute: false, encode: false, scale: false }
  },
  {
    templateId: 'scaled',
    typeId: 'wlearn.preprocess.tabular@1',
    params: { impute: 'auto', encode: 'onehot', scale: 'standard' }
  }
]
```

Templates are resolved before candidates are queued. V1 accepts fixed templates;
enumerate policies explicitly instead of putting a search space inside a template.
All five strategies cross model candidates with every template. When preprocessing
is active, stacking uses `passthrough: false`; explicitly requesting raw-feature
passthrough is rejected because fold-fitted feature spaces may differ.

Every model class or model spec must provide a stable `classId`. Candidate IDs are
opaque `wlc1_<sha256>` values derived from the class ID, exact model parameters,
and resolved preprocessing parameters. Display-name changes do not alter identity
or fold seeds. Use `candidate.model` and `candidate.preprocess`; do not parse the ID.

Refitted candidates are returned as Pipelines. Their WLRN artifacts retain the
structured candidate provenance, including when those Pipelines are nested in an
ensemble.

## Portfolio

The portfolio contains pre-tuned hyperparameter configs for these model families:

| Model | Package |
|-------|---------|
| xgb | `@wlearn/xgboost` |
| lgb | `@wlearn/lightgbm` |
| ebm | `@wlearn/ebm` |
| linear | `@wlearn/liblinear` |
| svm | `@wlearn/libsvm` |
| knn | `@wlearn/nanoflann` |
| tsetlin | `@wlearn/tsetlin` |
| rf | `@wlearn/rf` |
| mlp | `@wlearn/nn` (MLPClassifier/Regressor) |
| tabm | `@wlearn/nn` (TabMClassifier/Regressor) |
| nam | `@wlearn/nn` (NAMClassifier/Regressor) |
| gam | `@wlearn/gam` |
| bart | `@wlearn/stochtree` |
| fm | `@wlearn/xlearn` (FM) |
| xlr | `@wlearn/xlearn` (LR) |

Classification and regression have separate config sets with task-appropriate parameters.

## autoFit options

- `scoring` -- metric name (`'accuracy'`, `'r2'`, `'neg_mse'`) or custom function
- `cv` -- number of CV folds (default: 5)
- `nIter` -- number of search iterations for random/halving/progressive/Bayesian strategies (default: 20; BayesianSearch default: 30)
- `seed` -- random seed for reproducibility
- `task` -- `'classification'` or `'regression'` (auto-detected if omitted)
- `ensemble` -- build Caruana ensemble from top candidates (default: true)
- `ensembleSize` -- max ensemble members (default: 20)
- `refit` -- refit best model on full data (default: true)
- `preprocess` -- `false`, `true`, one preprocessing config, or fixed template list
- `strategy` -- `'random'`, `'portfolio'`, `'halving'`, `'progressive'`, or `'bayesian'`
- `stackingPassthrough` -- raw-feature passthrough; rejected with preprocessing

## Leaderboard

`result.leaderboard` is an array of `CandidateResult` objects sorted by score:

```js
{
  id: 0,
  candidateId: 'wlc1_...',
  candidate: {
    model: { displayName: 'xgb', classId: 'wlearn.xgboost.classifier@1', params: { max_depth: 6 } },
    preprocess: null
  },
  modelName: 'xgb',
  params: { max_depth: 6, eta: 0.1, ... },
  scores: Float64Array([0.92, 0.94, 0.91, 0.93, 0.90]),
  meanScore: 0.92,
  stdScore: 0.014,
  fitTimeMs: 42,
  rank: 1
}
```

## Archive

`result.archive` is an optional `Archive` from `@wlearn/core`. You do not need it for ordinary prediction or model selection. It is the agent/API-friendly record of the search:

```js
result.archive.records({ status: 'ok' })      // successful candidates
result.archive.records({ status: 'failed' })  // normalized failures
result.archive.leaderboard()                  // ranked rows from archive records
```

Each archive record includes params, budget, score, fit timing, fold scores, candidate id, and status. Keep using `result.leaderboard` for the compact ranked array; use `result.archive` when you need provenance, failure inspection, or a stable JSON-like run ledger.

## Testing

```bash
npm test             # unit/search/executor tests, no browser dependency
npm run test:browser # IIFE/ESM + real Tranfi-WASM preprocessing in Chromium
```

## License

Apache-2.0
