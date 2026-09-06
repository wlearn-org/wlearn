import * as sdk from '@wlearn/sdk'
const map = sdk.BasisTransformer.create({ method: 'rvfl', sampling: 'swim' })
void map
import type { EstimatorClass } from '@wlearn/types'

const modelClass: EstimatorClass = sdk.LinearModel
void modelClass

const task = sdk.createTask({
  id: 'sdk-types',
  kind: 'classification',
  X: [[0], [1]],
  y: new Int32Array([0, 1])
})
void sdk.taskRows(task)

const optimizerPromise = sdk.BayesianOptimizer.create({
  rate: { type: 'uniform', low: 0.01, high: 0.2 }
})
void optimizerPromise

if (sdk.MitraClassifier) {
  const classifier = sdk.MitraClassifier.create(
    new Uint8Array([1, 2, 3]),
    { maxSupport: 8, seed: 42 }
  )
  void classifier
}

// @ts-expect-error The unusable generic Mitra factory is not part of the SDK.
sdk.MitraModel
