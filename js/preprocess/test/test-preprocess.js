'use strict'

const assert = require('node:assert/strict')
const { before, test } = require('node:test')

const core = require('@wlearn/core')
const {
  MinMaxScaler,
  Preprocessor,
  StandardScaler,
  TYPE_ID,
  registerPreprocess,
  resolvePreprocessConfig
} = require('../src/index.js')

let tranfi = null
try {
  tranfi = require(process.env.TRANFI_JS_PATH || 'tranfi')
} catch (_) {
  // The package keeps Tranfi optional in the unreleased workspace test lane.
}
const skipWithoutTranfi = tranfi ? false : 'compatible Tranfi test backend is not installed'
const FINAL_TYPE_ID = 'wlearn.test.preprocess-final@1'

class FinalModel {
  constructor(cols = null) {
    this.cols = cols
    this.fitted = cols !== null
    this.disposed = false
  }

  fit(X) {
    this.cols = X.cols
    this.fitted = true
    return this
  }

  predict(X) {
    if (!this.fitted || this.disposed) throw new Error('final model is unavailable')
    return new Float64Array(X.rows).fill(this.cols)
  }

  save() {
    return core.encodeBundle({
      typeId: FINAL_TYPE_ID,
      params: { cols: this.cols }
    }, [])
  }

  getParams() { return { cols: this.cols } }
  setParams(params) { this.cols = params.cols; return this }
  dispose() { this.disposed = true }
  get capabilities() { return { regressor: true } }
}

core.register(
  FINAL_TYPE_ID,
  manifest => new FinalModel(manifest.params.cols),
  { sync: true }
)

before(async () => {
  if (tranfi) await registerPreprocess({ backend: tranfi })
})

test('re-exports core numeric scalers by exact identity', () => {
  assert.equal(StandardScaler, core.StandardScaler)
  assert.equal(MinMaxScaler, core.MinMaxScaler)
})

test('resolves configs without initializing the Tranfi backend', () => {
  const resolved = resolvePreprocessConfig({
    encode: 'label', scale: 'standard'
  })
  assert.equal(resolved.encode, 'label')
  assert.equal(resolved.scale, 'standard')
  assert.equal(resolved.policyVersion, 1)
  assert(Object.isFrozen(resolved))
  assert(Object.isFrozen(resolved.impute))
  assert.throws(
    () => resolvePreprocessConfig({ maxCategories: 1 }),
    /maxCategories/
  )
})

test('mixed inference, defaults, fit-transform, and zero-row apply', {
  skip: skipWithoutTranfi
}, async () => {
  const preprocessor = await Preprocessor.create()
  assert.deepEqual(preprocessor.getParams(), {
    impute: { numeric: 'mean', categorical: 'mode' },
    encode: 'onehot',
    scale: false,
    maxCategories: 20,
    unknownCategory: 'all_zero',
    allMissing: 'zero',
    maxOutputColumns: 65536,
    maxOutputElements: 100000000,
    policyVersion: 1
  })
  const output = preprocessor.fitTransform([
    [1, 10.5],
    [2, 20.5],
    [1, NaN],
    [2, 40.5]
  ])
  assert.deepEqual(
    [output.dtype, output.rows, output.cols],
    ['float64', 4, 3]
  )
  assert.deepEqual(Array.from(output.data), [
    1, 0, 10.5,
    0, 1, 20.5,
    1, 0, 23.833333333333336,
    0, 1, 40.5
  ])
  assert.deepEqual(
    preprocessor.outputSchema.map(field => field.role),
    ['onehot', 'onehot', 'value']
  )
  const unknown = preprocessor.transform([[3, 50.5]])
  assert.deepEqual(Array.from(unknown.data), [0, 0, 50.5])
  const empty = preprocessor.transform({
    dtype: 'float32', rows: 0, cols: 2, data: new Float32Array()
  })
  assert.deepEqual([empty.rows, empty.cols, empty.data.length], [0, 3, 0])
  preprocessor.dispose()
})

test('impute false preserves numeric NaN and label uses sentinel', {
  skip: skipWithoutTranfi
}, async () => {
  const passthrough = await Preprocessor.create({
    impute: false,
    encode: false
  })
  passthrough.fit([[1.5], [2.5]])
  const missing = passthrough.transform([[NaN]])
  assert(Number.isNaN(missing.data[0]))
  passthrough.dispose()

  const label = await Preprocessor.create({
    impute: false,
    encode: 'label'
  })
  label.fit([[1], [2], [1]])
  assert.deepEqual(Array.from(label.transform([[2], [3], [NaN]]).data), [1, -1, -1])
  label.dispose()
})

test('save/load validates plan identity and generic loader stays async', {
  skip: skipWithoutTranfi
}, async () => {
  const preprocessor = await Preprocessor.create({ scale: 'standard' })
  preprocessor.fit([[1, 10.5], [2, 20.5], [1, 30.5], [2, 40.5]])
  const bytes = preprocessor.save()
  const decoded = core.validateBundle(bytes)
  assert.equal(decoded.manifest.typeId, TYPE_ID)
  assert.match(decoded.manifest.metadata.tranfi.recipeSha256, /^[0-9a-f]{64}$/)

  let imports = 0
  const originalImport = tranfi.TransformPlan.fromBytes
  tranfi.TransformPlan.fromBytes = function(...args) {
    imports++
    return originalImport.apply(this, args)
  }
  try {
    assert.throws(() => core.loadSync(bytes), /Use async load/)
    assert.equal(imports, 0)
    const restored = await core.load(bytes, {
      loaderOptions: { [TYPE_ID]: {} }
    })
    assert.equal(imports, 1)
    assert.deepEqual(restored.getParams(), preprocessor.getParams())
    assert.deepEqual(
      Array.from(restored.transform([[1, 25.5]]).data),
      Array.from(preprocessor.transform([[1, 25.5]]).data)
    )
    restored.dispose()
  } finally {
    tranfi.TransformPlan.fromBytes = originalImport
    preprocessor.dispose()
  }
})

test('load rejects wrong outer type, metadata disagreement, and corrupt plan', {
  skip: skipWithoutTranfi
}, async () => {
  const wrong = core.encodeBundle(
    { typeId: 'wlearn.test.other@1' },
    [{ id: 'state', data: new Uint8Array([1]) }]
  )
  await assert.rejects(() => Preprocessor.load(wrong), /expected typeId/)

  const preprocessor = await Preprocessor.create()
  preprocessor.fit([[1], [2], [1]])
  const decoded = core.validateBundle(preprocessor.save())
  const plan = new Uint8Array(decoded.blobs)
  const metadata = structuredClone(decoded.manifest.metadata)
  metadata.outputSchema[0].name = 'tampered'
  const mismatched = core.encodeBundle({
    typeId: TYPE_ID,
    requires: [],
    params: decoded.manifest.params,
    metadata
  }, [{ id: 'plan', mediaType: decoded.toc[0].mediaType, data: plan }])
  await assert.rejects(
    () => Preprocessor.load(mismatched),
    error => error.code === 'ERR_BUNDLE' && /schemas do not match/.test(error.message)
  )

  const corruptPlan = new Uint8Array(plan)
  corruptPlan[corruptPlan.length - 1] ^= 1
  const corrupt = core.encodeBundle({
    typeId: TYPE_ID,
    requires: [],
    params: decoded.manifest.params,
    metadata: decoded.manifest.metadata
  }, [{ id: 'plan', mediaType: decoded.toc[0].mediaType, data: corruptPlan }])
  await assert.rejects(
    () => Preprocessor.load(corrupt),
    error => error.code === 'ERR_BUNDLE' &&
      error.engine === 'tranfi' && error.engineCode === 106 && Boolean(error.cause)
  )
  preprocessor.dispose()
})

test('load rejects boolean plan and policy version tags before import', {
  skip: skipWithoutTranfi
}, async () => {
  const preprocessor = await Preprocessor.create()
  preprocessor.fit([[1], [2], [1]])
  const decoded = core.validateBundle(preprocessor.save())
  const plan = new Uint8Array(decoded.blobs)
  let imports = 0
  const originalImport = tranfi.TransformPlan.fromBytes
  tranfi.TransformPlan.fromBytes = function(...args) {
    imports++
    return originalImport.apply(this, args)
  }
  try {
    for (const field of ['policyVersion', 'abiVersion', 'planFormatVersion']) {
      const params = structuredClone(decoded.manifest.params)
      const metadata = structuredClone(decoded.manifest.metadata)
      if (field === 'policyVersion') params.policyVersion = true
      else metadata.tranfi[field] = true
      const mutated = core.encodeBundle({
        typeId: TYPE_ID,
        requires: [],
        params,
        metadata
      }, [{ id: 'plan', mediaType: decoded.toc[0].mediaType, data: plan }])
      await assert.rejects(
        () => Preprocessor.load(mutated),
        error => error.code === (
          field === 'policyVersion' ? 'ERR_VALIDATION' : 'ERR_BUNDLE'
        )
      )
    }
    assert.equal(imports, 0)
  } finally {
    tranfi.TransformPlan.fromBytes = originalImport
    preprocessor.dispose()
  }
})

test('validation, resource limits, transactional fit, params, and disposal', {
  skip: skipWithoutTranfi
}, async () => {
  await assert.rejects(
    () => Preprocessor.create({ maxCategories: 1 }),
    error => error.code === 'ERR_VALIDATION'
  )
  await assert.rejects(
    () => Preprocessor.create(
      { maxCategories: 3 },
      { limits: { maxCategoriesPerColumn: 2 } }
    ),
    error => error.code === 'ERR_RESOURCE_LIMIT'
  )
  const preprocessor = await Preprocessor.create({ encode: false })
  preprocessor.fit([[1.5], [2.5]])
  assert.throws(() => preprocessor.fit([[Infinity]]), /finite numbers or NaN/)
  assert.deepEqual(Array.from(preprocessor.transform([[3.5]]).data), [3.5])
  assert.throws(
    () => preprocessor.transform({
      dtype: 'float64', rows: 1, cols: 1, data: new Float32Array([1])
    }),
    /dtype does not match/
  )
  assert.throws(() => preprocessor.transform([]), /declare its fitted width/)
  preprocessor.setParams({ scale: 'minmax' })
  assert.equal(preprocessor.isFitted, false)
  assert.throws(() => preprocessor.transform([[1.5]]), /not fitted/i)
  preprocessor.dispose()
  preprocessor.dispose()
  assert.throws(() => preprocessor.getParams(), /disposed/)
})

test('JS matches the shared cross-language plan SHA oracle', {
  skip: skipWithoutTranfi
}, async () => {
  const preprocessor = await Preprocessor.create({
    impute: 'median', encode: 'onehot', scale: 'minmax'
  })
  preprocessor.fit([
    [1, 4.5], [2, 1.5], [1, NaN], [2, 9.5]
  ])
  const decoded = core.validateBundle(preprocessor.save())
  assert.equal(
    core.sha256Sync(decoded.blobs),
    '28224e12cc56847f8cc2876cc96d38c561777b4831c3f6e3c13b5c0ff6cd8da2'
  )
  assert.equal(
    decoded.manifest.metadata.tranfi.recipeSha256,
    '5f151a8c4e97bf7966f8b801dc4d0f5ffc9445fea1224dd01599c7913cc924d0'
  )
  preprocessor.dispose()
})

test('nested Pipeline load forwards type-keyed runtime limits', {
  skip: skipWithoutTranfi
}, async () => {
  const preprocessor = await Preprocessor.create({
    impute: 'median', encode: 'label', scale: 'minmax'
  })
  const pipeline = new core.Pipeline([
    ['preprocess', preprocessor],
    ['model', new FinalModel()]
  ])
  pipeline.fit(
    [[1, 4.5], [2, 1.5], [1, NaN], [2, 9.5]],
    new Float64Array([0, 1, 0, 1])
  )
  const bytes = pipeline.save()
  pipeline.dispose()
  assert.throws(() => core.loadSync(bytes), /Use async load/)

  const restored = await core.load(bytes, {
    loaderOptions: {
      [TYPE_ID]: { limits: { maxApplyRows: 2 } }
    }
  })
  assert.deepEqual(Array.from(restored.predict([[2, 4.5], [3, 5.5]])), [2, 2])
  assert.throws(
    () => restored.predict([[1, 1], [2, 2], [3, 3]]),
    error => error.code === 'ERR_RESOURCE_LIMIT'
  )
  restored.dispose()
})

test('cancellation and unsupported runtime preserve fitted-plan ownership', {
  skip: skipWithoutTranfi
}, async () => {
  const cancelFlag = new Int32Array(new SharedArrayBuffer(4))
  const preprocessor = await Preprocessor.create(
    { encode: false }, { cancelFlag }
  )
  preprocessor.fit([[1.5], [2.5]])
  Atomics.store(cancelFlag, 0, 1)
  assert.throws(
    () => preprocessor.transform([[3.5]]),
    error => error.code === 'ERR_CANCELLED' &&
      error.engine === 'tranfi' && error.engineCode === 109 && Boolean(error.cause)
  )
  assert.equal(preprocessor.isFitted, true)
  Atomics.store(cancelFlag, 0, 0)
  assert.deepEqual(Array.from(preprocessor.transform([[3.5]]).data), [3.5])

  const original = tranfi.TransformRecipe.fromJSON
  tranfi.TransformRecipe.fromJSON = function() {
    const error = new Error('unsupported floating-point runtime')
    error.code = 113
    throw error
  }
  try {
    assert.throws(
      () => preprocessor.fit([[4.5]]),
      error => error.code === 'ERR_BACKEND' &&
        error.engine === 'tranfi' && error.engineCode === 113 && Boolean(error.cause)
    )
  } finally {
    tranfi.TransformRecipe.fromJSON = original
  }
  assert.equal(preprocessor.isFitted, true)
  assert.deepEqual(Array.from(preprocessor.transform([[4.5]]).data), [4.5])
  preprocessor.dispose()
})
