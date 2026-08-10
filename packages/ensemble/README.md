# @wlearn/ensemble

Ensemble methods for wlearn: voting, stacking, bagging, and Caruana greedy selection.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/ensemble
```

## Voting

Combine multiple models by averaging predictions (soft) or majority vote (hard).

```js
const { VotingEnsemble } = require('@wlearn/ensemble')
const { LinearModel } = require('@wlearn/liblinear')
const { XGBModel } = require('@wlearn/xgboost')

const ens = await VotingEnsemble.create({
  estimators: [
    ['linear', LinearModel, { task: 'classification' }],
    ['xgb', XGBModel, { task: 'classification' }]
  ],
  voting: 'soft',
  task: 'classification'
})

ens.fit(X, y)
const preds = ens.predict(X_test)
const acc = ens.score(X_test, y_test)
```

## Stacking

Train base models, collect out-of-fold predictions, then train a meta-learner on those predictions.

```js
const { StackingEnsemble } = require('@wlearn/ensemble')
const { LinearModel } = require('@wlearn/liblinear')
const { XGBModel } = require('@wlearn/xgboost')

const stack = await StackingEnsemble.create({
  estimators: [
    ['xgb', XGBModel, { task: 'classification' }],
    ['linear', LinearModel, { task: 'classification' }]
  ],
  finalEstimator: ['meta', LinearModel, { task: 'classification' }],
  cv: 5,
  task: 'classification'
})

stack.fit(X, y)
const preds = stack.predict(X_test)
```

## Bagging

Bootstrap aggregating: train multiple copies of the same model on bootstrap samples.

```js
const { BaggedEstimator } = require('@wlearn/ensemble')
const { LinearModel } = require('@wlearn/liblinear')

const bag = await BaggedEstimator.create({
  estimator: ['linear', LinearModel, { task: 'classification' }],
  nEstimators: 10,
  task: 'classification'
})

bag.fit(X, y)
const preds = bag.predict(X_test)
```

## API

- `VotingEnsemble.create(opts)` supports soft/hard voting over fitted submodels.
- `StackingEnsemble.create(opts)` trains base models and a final estimator on out-of-fold predictions.
- `BaggedEstimator.create(opts)` trains bootstrap copies of one estimator spec.
- Ensemble instances implement `fit(X, y)`, `predict(X)`, `predictProba(X)`, `score(X, y)`, `save()`, and `dispose()` for deterministic cleanup in long-running loops.
- Estimator specs use `[name, ModelClass, params]`.

## Utilities

- `caruanaSelect(oofPredictions, yTrue, opts?)` -- Caruana greedy ensemble selection
- `getOofPredictions(estimatorSpecs, X, y, opts?)` -- compute out-of-fold predictions
- `optimizeWeights(oofPredictions, yTrue, opts?)` -- optimize ensemble weights
- `projectSimplex(weights)` -- project weights onto probability simplex

## Testing

```bash
npm test             # ensemble unit tests, no browser dependency
npm run test:browser # builds IIFE/ESM bundles and checks exports in Chromium
```

## License

Apache-2.0
