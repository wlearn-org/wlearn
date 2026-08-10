'use strict'

const assert = require('node:assert/strict')
const { test } = require('node:test')

const {
  MinMaxScaler,
  Preprocessor,
  StandardScaler,
  registerPreprocess
} = require('../src/wasm.js')
const core = require('@wlearn/core')

let createTranfi = null
try {
  createTranfi = require(process.env.TRANFI_WASM_PATH || 'tranfi/wasm')
} catch (_) {
  // The package keeps Tranfi optional in the unreleased workspace test lane.
}

test('standalone WASM entry re-exports core scalers by exact identity', () => {
  assert.equal(StandardScaler, core.StandardScaler)
  assert.equal(MinMaxScaler, core.MinMaxScaler)
})

test('standalone WASM fits, saves, imports, and applies the same contract', {
  skip: createTranfi ? false : 'compatible Tranfi WASM test backend is not installed'
}, async () => {
  const backend = await createTranfi()
  await registerPreprocess({ backend })
  const preprocessor = await Preprocessor.create({
    impute: 'median',
    encode: 'label',
    scale: 'minmax'
  })
  const fitted = preprocessor.fitTransform([
    [1, 4.5],
    [2, 1.5],
    [1, NaN],
    [2, 9.5]
  ])
  assert.deepEqual([fitted.rows, fitted.cols], [4, 2])
  assert.deepEqual(Array.from(fitted.data), [
    0, 0.375,
    1, 0,
    0, 0.375,
    1, 1
  ])
  const bundle = preprocessor.save()
  const restored = await Preprocessor.load(bundle)
  assert.equal(restored.backend, 'wasm')
  assert.deepEqual(
    Array.from(restored.transform([[2, 4.5], [3, 5.5]]).data),
    [1, 0.375, -1, 0.5]
  )
  restored.dispose()
  preprocessor.dispose()
})
