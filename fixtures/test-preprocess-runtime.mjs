import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import test from 'node:test'

// Resolve from the consumer to run the same contract against packed installs.
const require = createRequire(resolve('package.json'))
const { Preprocessor, registerPreprocess } = require('@wlearn/preprocess')
const { load, decodeBundle } = require('@wlearn/core')
const { LinearModel } = require('@wlearn/liblinear')
const { autoFit } = require('@wlearn/automl')
const tranfi = require('tranfi')
const expectedBackend = tranfi.hasNativePreparedTransforms() ? 'native' : 'wasm'
const config = {
  impute: 'auto', encode: 'onehot', scale: 'standard',
  columns: { x0: { kind: 'numeric' }, x1: { kind: 'categorical' }, x2: { kind: 'numeric' } }
}
const X = Array.from({ length: 48 }, (_, i) => [(i - 24) / 7, i % 3, i % 7 === 0 ? NaN : (i % 9) / 4])
const y = Int32Array.from(X, row => row[0] > 0 ? 1 : 0)

test('default preprocessing resolves its backend and persists through core.load', async () => {
  const prep = await Preprocessor.create(config)
  try {
    assert.equal(prep.backend, expectedBackend)
    const fitted = prep.fit(X, y)
    assert.equal(fitted, prep)
    const out = prep.transform(X)
    assert.equal(out.cols, 5)
    assert.ok(Array.from(out.data).every(Number.isFinite))
    const restored = await load(prep.save())
    try { assert.deepEqual(restored.transform(X), out) } finally { restored.dispose() }
  } finally { prep.dispose() }
})

test('AutoML default ensemble persists preprocessing without a native requirement', async () => {
  const models = [{ name: 'linear', classId: 'wlearn.liblinear.classifier@1', cls: LinearModel, params: { task: 'classification' } }]
  const result = await autoFit(models, X, y, { cv: 2, nIter: 3, seed: 42, preprocess: config })
  try {
    const expected = await result.model.predict(X)
    const bytes = result.model.save()
    assert.match(decodeBundle(bytes).manifest.typeId, /ensemble/)
    const restored = await load(bytes)
    try { assert.deepEqual(await restored.predict(X), expected) } finally { restored.dispose() }
  } finally { result.model.dispose() }
})

test('selected backend keeps its cancellation and wrong-option contracts', async () => {
  const backend = await registerPreprocess()
  const flag = new Int32Array(new SharedArrayBuffer(4))
  const token = expectedBackend === 'wasm' ? backend.createTransformCancelToken() : null
  const prep = await Preprocessor.create(config, token ? { cancelToken: token } : { cancelFlag: flag })
  try {
    if (token) token.request()
    else Atomics.store(flag, 0, 1)
    assert.throws(() => prep.fit(X, y), error => error.name === 'CancelledError')
    await assert.rejects(() => Preprocessor.create(config, token ? { cancelFlag: flag } : { cancelToken: {} }), /backend uses/)
    if (token) assert.equal(Preprocessor, require('@wlearn/preprocess/wasm').Preprocessor)
  } finally { prep.dispose(); token?.close() }
})
