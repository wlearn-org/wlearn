import {
  Pipeline,
  StandardScaler,
  decodeBundle,
  encodeBundle,
  load,
  loadSync,
  normalizeX,
  register
} from '@wlearn/core'
import type { MaybePromise } from '@wlearn/types'

const matrix = normalizeX([[1], [2]])
const scaler = new StandardScaler().fit(matrix)
const pipeline = new Pipeline([['scale', scaler], ['model', scaler]])
const pipelineFit: MaybePromise<Pipeline> = pipeline.fit(
  matrix, new Int32Array([0, 1])
)
const pipelineClasses: Int32Array | null = pipeline.classes
const bytes = encodeBundle({ typeId: 'wlearn.test.types@1' }, [])
decodeBundle(bytes)
register('wlearn.test.types@1', () => scaler, { sync: true })
const sync = loadSync<typeof scaler>(bytes)
const pending: Promise<typeof scaler> = load<typeof scaler>(bytes)
void pipeline
void pipelineFit
void pipelineClasses
void sync
void pending
