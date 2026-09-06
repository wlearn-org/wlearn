import * as wlearnTypes from '@wlearn/types'
import type {
  Archive,
  Estimator,
  MaybePromise,
  Pipeline,
  Task
} from '@wlearn/types'

const version: 1 = wlearnTypes.BUNDLE_VERSION
const magic: Uint8Array = wlearnTypes.BUNDLE_MAGIC

// Runtime implementations belong to @wlearn/core, not @wlearn/types.
// @ts-expect-error Pipeline is a contract type, not a types-package value.
new wlearnTypes.Pipeline([])
// @ts-expect-error accuracy is implemented and declared by @wlearn/core.
wlearnTypes.accuracy(new Int32Array(), new Int32Array())

declare const estimator: Estimator
declare const pipeline: Pipeline
declare const archive: Archive
declare const task: Task
const prediction: MaybePromise<Int32Array | Float32Array | Float64Array> =
  estimator.predict(task.X)

void version
void magic
void pipeline
void archive
void prediction
