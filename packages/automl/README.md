# @wlearn/automl

Automated model selection for wlearn. Searches over model families and hyperparameters, runs cross-validation, and optionally builds a Caruana ensemble from top candidates.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

> **Unreleased main:** This README targets AutoML 0.3. The current npm 0.2.x
> release does not include the preprocessing integration described below, and
> `@wlearn/preprocess` is not yet published. Use the coordinated source trees for
> that workflow until the next release.

## Install

```bash
npm install @wlearn/automl @wlearn/liblinear
```

AutoML orchestrates model packages; it does not contain a model backend itself.
The command above includes the backend used by this example.

## Quick start

```js
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')

const models = [
  { name: 'linear', classId: 'wlearn.liblinear.classifier@1',
    portfolioKey: 'linear', cls: LinearModel, params: { task: 'classification' } }
]

async function main() {
  const X = [[-4], [-3], [-2], [-1], [1], [2], [3], [4]]
  const y = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1])

  const result = await autoFit(models, X, y, {
    scoring: 'accuracy',
    cv: 2,
    nIter: 2,
    ensemble: false,
    refit: true,
    seed: 42
  })

  console.log(result.bestScore) // 1
  console.log(Array.from(result.model.predict([[-2], [2]]))) // [0, 1]
  console.log(result.leaderboard.length) // 2
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
```

For the simple path, this is the whole API: use `result.model` for prediction and
`result.leaderboard` for ranked candidates. The LinearModel winner above predicts
synchronously; a winner backed by an asynchronous runtime returns a Promise from
prediction. `result.archive` is optional metadata for provenance, failed
candidates, and agent/API workflows.

Add more model packages and model specs to compare families. In long-running
search services, call `result.model.dispose()` after the selected model is no
longer needed; AutoML disposes rejected fold candidates itself.

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

`baseSeed` is the run seed. `foldSeeds` are deterministic values derived from the
structured candidate identity, fold index, and `baseSeed`; the executor records
them for provenance and uses them for executor-owned random operations such as a
successive-halving subsample budget. They do not replace the model's explicit
candidate `seed` parameter and do not describe how the already-supplied CV folds
were generated.

When preprocessing is active, refitted candidates are returned as Pipelines.
Those Pipeline WLRN artifacts retain the structured candidate provenance,
including when they are nested in an ensemble. Without preprocessing, refit
returns the fitted base model directly.

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
  baseSeed: 42,
  foldSeeds: Uint32Array([/* one derived seed per fold */]),
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
