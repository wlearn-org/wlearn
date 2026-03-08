# @wlearn/automl

Automated model selection for wlearn. Searches over model families and hyperparameters, runs cross-validation, and optionally builds a Caruana ensemble from top candidates.

Part of the [wlearn](https://github.com/wlearn-org/wlearn) project.

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
  ['linear', LinearModel, { task: 'classification' }],
  ['xgb', XGBModel, { task: 'classification' }]
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
result.bestScore     // best CV score
result.bestModelName // e.g. 'xgb'
result.bestParams    // winning hyperparameters

result.model.dispose()
```

## Search strategies

- `RandomSearch` -- random hyperparameter sampling (default)
- `SuccessiveHalvingSearch` -- early stopping with increasing resource allocation
- `PortfolioSearch` -- predefined portfolio of known-good configurations
- `ProgressiveSearch` -- progressive resource allocation

Each model provides a default search space via `Model.defaultSearchSpace()`. AutoML samples from these automatically.

## autoFit options

- `scoring` -- metric name (`'accuracy'`, `'r2'`, `'neg_mse'`) or custom function
- `cv` -- number of CV folds (default: 5)
- `nIter` -- number of random search iterations (default: 10)
- `seed` -- random seed for reproducibility
- `task` -- `'classification'` or `'regression'` (auto-detected if omitted)
- `ensemble` -- build Caruana ensemble from top candidates (default: false)
- `ensembleSize` -- max ensemble members (default: 20)
- `refit` -- refit best model on full data (default: true)

## Leaderboard

`result.leaderboard` is an array of `CandidateResult` objects sorted by score:

```js
{
  id: 0,
  modelName: 'xgb',
  params: { max_depth: 6, eta: 0.1, ... },
  scores: Float64Array([0.92, 0.94, 0.91, 0.93, 0.90]),
  meanScore: 0.92,
  stdScore: 0.014,
  fitTimeMs: 42,
  rank: 1
}
```

## License

MIT
