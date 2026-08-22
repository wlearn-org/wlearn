# @wlearn/ensemble

Ensemble methods for wlearn: voting, stacking, bagging, and Caruana greedy selection.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)).

## Install

```bash
npm install @wlearn/ensemble @wlearn/liblinear
```

Ensemble constructors receive model classes; install every model package named by
your estimator specs. The command above includes the backend used below.

## Voting

Combine multiple models by averaging predictions (soft) or majority vote (hard).

```js
const { VotingEnsemble } = require('@wlearn/ensemble')
const { LinearModel } = require('@wlearn/liblinear')

async function main() {
  const X = [[-4], [-3], [-2], [-1], [1], [2], [3], [4]]
  const y = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1])

  const ensemble = await VotingEnsemble.create({
    estimators: [
      ['linear-c1', LinearModel, { task: 'classification', C: 1 }],
      ['linear-c2', LinearModel, { task: 'classification', C: 2 }]
    ],
    voting: 'hard',
    task: 'classification'
  })

  await ensemble.fit(X, y)
  // Inference stays synchronous because both fitted children are synchronous.
  console.log(Array.from(ensemble.predict([[-2], [2]]))) // [0, 1]
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
```

## Stacking

Train base models, collect out-of-fold predictions, then train a meta-learner on those predictions.

The following focused snippets assume `X`, `y`, and `XTest` are defined inside an
async function and that both model packages are installed.

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

await stack.fit(X, y)
const preds = stack.predict(XTest)
```

## K-fold bagging

Train one model per held-out fold and retain averaged out-of-fold (OOF)
predictions. Repeats use new deterministic fold assignments; this is repeated
K-fold bagging, not bootstrap sampling.

```js
const { BaggedEstimator } = require('@wlearn/ensemble')
const { LinearModel } = require('@wlearn/liblinear')

const bag = await BaggedEstimator.create({
  estimator: ['linear', LinearModel, { task: 'classification' }],
  kFold: 5,
  nRepeats: 2,
  task: 'classification'
})

await bag.fit(X, y)
const preds = bag.predict(XTest)
const oof = bag.oofPredictions
```

A fitted `BaggedEstimator` can be supplied to stacking as
`['baggedName', bag]`. Stacking consumes its OOF predictions without retraining
it and takes ownership only after the stacking fit commits successfully. Its task,
class set, and exact OOF row/column shape must match the stacking data; stored
probability columns must be finite and are aligned from its declared class order.
Regression child predictions must likewise contain exactly one finite numeric
value per input row at OOF, aggregation, meta-feature, and final-output boundaries.
Legacy bagging artifacts that omit OOF data remain loadable for inference, but
their OOF accessor, canonical re-save, and use as a pre-fitted stacking base are
rejected rather than treating missing training evidence as zeros.

## API

- `VotingEnsemble.create(opts)` supports soft/hard voting over fitted submodels.
- `StackingEnsemble.create(opts)` trains base models and a final estimator on out-of-fold predictions.
- `BaggedEstimator.create(opts)` trains `kFold * nRepeats` copies of one estimator spec.
- Ensemble instances implement asynchronous `fit(X, y)`, MaybePromise inference, `save()`, and `dispose()` for deterministic cleanup in long-running loops.
- Estimator specs use `[name, ModelClass, params]`; stacking also accepts `[name, fittedBaggedEstimator]`.

Fit/refit is transactional: a failed replacement is cleaned up without changing
the previous fitted state. After a successful commit, cleanup failures from old
models do not turn the completed fit into a rejection. Bundle loaders validate
ensemble task, names, weights, class metadata, artifact IDs/media types, and OOF
shape before dispatching any child loader.
While asynchronous child construction or fit is pending, a second `fit()`,
`setParams()`, or `dispose()` call is rejected. Await fit completion before any of
those lifecycle operations.

Changing a bagging or stacking training parameter with `setParams()` invalidates
the fitted state; call `fit()` again before inference or persistence. Bagging and
stacking validate task, fold/repeat counts, seeds, names, constructors, and
passthrough configuration before creating children, and a rejected `setParams()`
update leaves the prior fitted state and configuration intact.
Mutable fields are exact: Voting accepts only `voting`/`weights`, Bagging only
`kFold`/`nRepeats`/`seed`, and Stacking only `cv`/`passthrough`/`seed`; unknown or
constructor-only fields are rejected instead of becoming silent no-ops.
Voting validates nonempty unique estimator specs and nonnegative finite weights
with a positive sum before training. Weights are relative and normalized to sum
to one. An invalid inference-only `setParams()` update is rejected without
changing a fitted ensemble. Children used for probability aggregation must expose
their fitted `classes`; soft voting also requires an explicit
`capabilities.predictProba === true` declaration. Pipelines forward the final
estimator's classes and capabilities, so a pipeline ending in a probability
classifier remains a valid soft-voting child. Soft voting, bagging, and stacking
validate the class set and align every `predictProba` block to the ensemble class
order. Stacking derives its own probability capability from the fitted meta-model.
Hard voting needs only label predictions, calls each child once per inference, and
rejects wrong-shape, non-finite, fractional, or unknown class labels.
Classification `predict()` results are `Int32Array` in JavaScript and NumPy
`int32` in Python; regression remains float64.

## Utilities

- `caruanaSelect(oofPredictions, yTrue, opts?)` -- Caruana greedy ensemble selection; pass `opts.classes` when probability columns use an explicit class order
- `getOofPredictions(estimatorSpecs, X, y, opts?)` -- compute out-of-fold predictions
- `optimizeWeights(oofPredictions, yTrue, initialWeights, opts?)` -- optimize ensemble weights
- `projectSimplex(weights)` -- project weights onto probability simplex

## Testing

```bash
npm test             # ensemble unit tests, no browser dependency
npm run test:types   # compile public TypeScript usage
npm run test:browser # builds IIFE/ESM bundles and checks exports in Chromium
```

## License

Apache-2.0
