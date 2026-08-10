// wlearn bundle format constants
const BUNDLE_MAGIC = new Uint8Array([0x57, 0x4c, 0x52, 0x4e]) // 'WLRN'
const BUNDLE_VERSION = 1
const HEADER_SIZE = 16 // magic(4) + version(4) + manifestLen(4) + tocLen(4)
const DTYPE = { FLOAT32: 'float32', FLOAT64: 'float64', INT32: 'int32' }
const TASK_KINDS = [
  'classification',
  'regression',
  'clustering',
  'ranking',
  'survival',
  'forecasting',
  'multioutput',
  'anomaly'
]
const PREDICTION_FIELDS = ['response', 'proba', 'score', 'decision', 'interval', 'quantiles']
const MEASURE_DIRECTIONS = ['maximize', 'minimize']
const MEASURE_RESPONSES = ['response', 'proba', 'score', 'decision', 'distribution']
const RESAMPLING_STRATEGIES = [
  'holdout',
  'kfold',
  'stratified_kfold',
  'repeated_kfold',
  'group_kfold',
  'time_series',
  'sliding_window',
  'sliding_index',
  'sliding_period'
]
const TRIAL_STATUSES = ['pending', 'running', 'ok', 'failed', 'pruned', 'timeout']

module.exports = {
  BUNDLE_MAGIC,
  BUNDLE_VERSION,
  HEADER_SIZE,
  DTYPE,
  TASK_KINDS,
  PREDICTION_FIELDS,
  MEASURE_DIRECTIONS,
  MEASURE_RESPONSES,
  RESAMPLING_STRATEGIES,
  TRIAL_STATUSES
}
