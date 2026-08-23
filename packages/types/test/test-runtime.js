const { test } = require('node:test')
const assert = require('node:assert/strict')

const types = require('..')

test('@wlearn/types runtime exports only its declared constants', () => {
  assert.deepEqual(Object.keys(types).sort(), [
    'BUNDLE_MAGIC',
    'BUNDLE_VERSION',
    'DTYPE',
    'HEADER_SIZE',
    'MEASURE_DIRECTIONS',
    'MEASURE_RESPONSES',
    'PREDICTION_FIELDS',
    'RESAMPLING_STRATEGIES',
    'TASK_KINDS',
    'TRIAL_STATUSES'
  ])
})
