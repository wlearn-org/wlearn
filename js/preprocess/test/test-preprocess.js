'use strict'

const assert = require('node:assert/strict')
const { test } = require('node:test')

const core = require('@wlearn/core')
const {
  MinMaxScaler,
  Preprocessor,
  StandardScaler,
  TYPE_ID,
  registerPreprocess,
  resolvePreprocessConfig
} = require('../src/index.js')

async function cancellation(t) {
  const tranfi = await registerPreprocess()
  const flag = new Int32Array(new SharedArrayBuffer(4))
  // Exercise the selected entry's contract; WASM tokens can poll a shared flag.
  const token = typeof tranfi.createTransformCancelToken === 'function'
    ? tranfi.createTransformCancelToken({ sharedFlag: flag })
    : null
  if (token) t.after(() => token.close())
  return {
    options: token ? { cancelToken: token } : { cancelFlag: flag },
    set: value => Atomics.store(flag, 0, value)
  }
}
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

test('mixed inference, defaults, fit-transform, and zero-row apply', async () => {
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

test('impute false preserves numeric NaN and label uses sentinel', async () => {
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

test('save/load validates plan identity and generic loader stays async', async () => {
  const tranfi = await registerPreprocess()
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

test('load rejects wrong outer type, metadata disagreement, and corrupt plan', async () => {
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

test('load rejects boolean plan and policy version tags before import', async () => {
  const tranfi = await registerPreprocess()
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

test('validation, resource limits, transactional fit, params, and disposal', async () => {
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

test('JS matches the shared cross-language plan SHA oracle', async () => {
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

test('nested Pipeline load forwards type-keyed runtime limits', async () => {
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

test('cancellation and unsupported runtime preserve fitted-plan ownership', async t => {
  const tranfi = await registerPreprocess()
  const cancel = await cancellation(t)
  const preprocessor = await Preprocessor.create(
    { encode: false }, cancel.options
  )
  preprocessor.fit([[1.5], [2.5]])
  cancel.set(1)
  assert.throws(
    () => preprocessor.transform([[3.5]]),
    error => error.code === 'ERR_CANCELLED' &&
      error.engine === 'tranfi' && error.engineCode === 109 && Boolean(error.cause)
  )
  assert.equal(preprocessor.isFitted, true)
  cancel.set(0)
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

test('pre-cancelled preprocessing does not read matrix values', async t => {
  const cancel = await cancellation(t)
  const processor = await Preprocessor.create(
    { impute: false, encode: false, scale: false }, cancel.options
  )
  processor.fit([[1, 2], [3, 4]])
  cancel.set(1)
  const row = [1, 2]
  Object.defineProperty(row, 0, { get() { throw new Error('cancelled input was read') } })
  for (const operation of ['fit', 'transform']) {
    assert.throws(() => processor[operation]([row]), error => error.engineCode === 109)
  }
  processor.dispose()
})

test('cancellation interrupts matrix conversion', async t => {
  const cancel = await cancellation(t)
  const processor = await Preprocessor.create(
    { impute: false, encode: false, scale: false }, cancel.options
  )
  processor.fit([[1]])
  let reads = 0
  const row = [1]
  Object.defineProperty(row, 0, { get() {
    reads++
    cancel.set(1)
    return 1
  } })
  assert.throws(() => processor.transform(Array(30000).fill(row)),
    error => error.engineCode === 109)
  assert(reads > 0 && reads <= 8192)
  processor.dispose()
})

test('WASM cancellation and returned buffer ownership', async () => {
  const path = require('node:path')
  const createTranfi = process.env.TRANFI_JS_PATH
    ? require(path.join(process.env.TRANFI_JS_PATH, 'wasm'))
    : require('tranfi/wasm')
  const backend = await createTranfi()
  const { createPreprocessAPI } = require('../src/factory.js')
  const api = createPreprocessAPI(() => backend, 'wasm')
  const cancelToken = backend.createTransformCancelToken()
  const processor = await api.Preprocessor.create(
    { impute: false, encode: false, scale: false }, { cancelToken }
  )
  processor.fit([[1], [2]])
  const result = processor.transform([[3], [4]])
  const second = processor.transform([[5], [6]])
  assert.equal(cancelToken.requested, false)
  cancelToken.request()
  assert.equal(cancelToken.requested, true)
  const row = [1]
  Object.defineProperty(row, 0, { get() { throw new Error('cancelled input was read') } })
  assert.throws(() => processor.transform([row]), error => error.engineCode === 109)
  processor.dispose()
  cancelToken.close()
  assert.deepEqual([...result.data], [3, 4])
  result.data[0] = 99
  assert.deepEqual([...second.data], [5, 6])
})

test('column policies preserve fixed widths and explicit kinds', async () => {
  const pre = await Preprocessor.create({ scale: 'standard', columns: {
    x0: { kind: 'numeric', scale: false },
    x1: { categories: [5, 0, 2] },
    x2: { kind: 'categorical', encode: 'label' }
  } })
  pre.fit([[1, 2, 7], [2, 2, 8]])
  assert.deepEqual(Array.from(pre.transform([[3, 5, 8], [4, 99, 9]]).data),
    [3, 0, 0, 1, 1, 4, 0, 0, 0, -1])
  const saved = pre.save()
  const restored = await Preprocessor.load(saved)
  assert.deepEqual(restored.save(), saved)
  assert.deepEqual(restored.getParams().columns.x1.categories, [0, 2, 5])
  pre.fit([[3, 0, 7], [4, 5, 8]])
  assert.equal(pre.outputSchema.length, 5)
  pre.setParams({ scale: 'minmax' })
  assert.equal(pre.isFitted, false)
  pre.fit([[1, 2, 7], [2, 2, 8]])
  assert.equal(pre.transform([[3, 5, 8]]).data[0], 3)
  pre.setParams({ columns: {} })
  assert.equal(Object.hasOwn(pre.getParams(), 'columns'), false)
  pre.dispose()
  restored.dispose()
})

test('validates column policies without a backend', () => {
  for (const columns of [
    { x01: {} }, { 'x-1': {} }, { x0: { kind: 'string' } },
    { x0: { categories: [] } }, { x0: { categories: [1, 1] } },
    { x0: { categories: [true] } }, { x0: { categories: [Infinity] } },
    { x0: { kind: 'infer', categories: [0, 1] } },
    { x0: { maxOutputColumns: 2 } }
  ]) assert.throws(() => resolvePreprocessConfig({ columns }), core.ValidationError)
  assert.throws(() => resolvePreprocessConfig({ impute: false, encode: false,
    columns: { x0: { categories: [0, 1] } } }), core.ValidationError)
})

test('column overrides inherit global updates and reject invalid patches atomically', async () => {
  const pre = await Preprocessor.create({ columns: { x0: { kind: 'numeric' } } })
  pre.fit([[1], [3]])
  const original = pre.save()
  assert.throws(() => pre.setParams({ columns: { x0: { categories: [1, 1] } } }), core.ValidationError)
  assert.deepEqual(pre.save(), original)
  pre.setParams({ scale: 'minmax' })
  assert.deepEqual(Array.from(pre.fitTransform([[1], [3]]).data), [0, 1])
  pre.setParams({ columns: { x1: { kind: 'numeric' } } })
  assert.throws(() => pre.fit([[1], [3]]), /x1/)
  pre.dispose()
})

test('column policies respect host limits before fit', async () => {
  const limits = { maxCategoriesPerColumn: 2, maxTotalCategories: 3 }
  for (const columns of [
    { x0: { categories: [0, 1, 2] } },
    { x0: { categories: [0, 1] }, x1: { categories: [0, 1] } },
    { x0: { maxCategories: 3 } }
  ]) await assert.rejects(Preprocessor.create({ maxCategories: 2, columns }, { limits }), core.ResourceLimitError)
})

test('column inference threshold and imputation overrides', async () => {
  const pre = await Preprocessor.create({ columns: {
    x0: { maxCategories: 2, impute: 'median' },
    x1: { kind: 'categorical', impute: false, encode: 'label' }
  } })
  pre.fit([[1, 0.5], [2, 1.5], [9, 0.5]])
  assert.deepEqual(Array.from(pre.transform([[NaN, NaN], [3, 1.5]]).data), [2, -1, 3, 1])
  pre.dispose()
})
