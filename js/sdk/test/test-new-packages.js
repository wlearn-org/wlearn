const { test } = require('node:test')
const assert = require('node:assert/strict')
const sdk = require('@wlearn/sdk')

test('SDK exposes Sym and uncertainty with the shared bundle loader', async () => {
  assert.equal(typeof sdk.SymbolicRegressor?.create, 'function')
  assert.equal(typeof sdk.FormulaTransformer?.create, 'function')
  const calibrator = await sdk.IntervalCalibrator.create()
  let restored
  try {
    calibrator.fit([0, 0, 0, 0], [-2, -1, 1, 2])
    restored = await sdk.load(calibrator.save())
    assert.deepEqual(restored.predictInterval([0], [.8]), calibrator.predictInterval([0], [.8]))
  } finally {
    restored?.dispose()
    calibrator.dispose()
  }
})
