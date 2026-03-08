const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { Pipeline } = require('../src/pipeline.js')
const { register } = require('../src/registry.js')
const { encodeBundle, decodeBundle } = require('../src/bundle.js')
const { DisposedError, NotFittedError, ValidationError } = require('../src/errors.js')

// Mock transformer: doubles all values
function createMockTransformer(name) {
  let fitted = false
  let disposed = false
  return {
    fit(X, y) { fitted = true; return this },
    transform(X) {
      const data = new Float64Array(X.data.length)
      for (let i = 0; i < data.length; i++) data[i] = X.data[i] * 2
      return { data, rows: X.rows, cols: X.cols }
    },
    fitTransform(X, y) { this.fit(X, y); return this.transform(X) },
    predict(X) { return new Float64Array(X.rows) },
    score(X, y) { return 0.5 },
    save() {
      return encodeBundle(
        { typeId: `wlearn.mock.transformer.${name}@1` },
        [{ id: 'state', data: new Uint8Array([1]) }]
      )
    },
    dispose() { disposed = true },
    getParams() { return { name } },
    setParams(p) { return this },
    get capabilities() {
      return { classifier: false, regressor: true, predictProba: false, decisionFunction: false, sampleWeight: false, csr: false, earlyStopping: false }
    },
    get isFitted() { return fitted },
    get isDisposed() { return disposed }
  }
}

// Mock classifier (last step)
function createMockClassifier() {
  let fitted = false
  let disposed = false
  return {
    fit(X, y) { fitted = true; return this },
    predict(X) {
      const out = new Float64Array(X.rows)
      for (let i = 0; i < out.length; i++) out[i] = i % 2
      return out
    },
    predictProba(X) {
      const out = new Float64Array(X.rows * 2)
      for (let i = 0; i < X.rows; i++) {
        out[i * 2] = 0.3
        out[i * 2 + 1] = 0.7
      }
      return out
    },
    score(X, y) { return 0.9 },
    save() {
      return encodeBundle(
        { typeId: 'wlearn.mock.classifier@1' },
        [{ id: 'state', data: new Uint8Array([2]) }]
      )
    },
    dispose() { disposed = true },
    getParams() { return { type: 'classifier' } },
    setParams(p) { return this },
    get capabilities() {
      return { classifier: true, regressor: false, predictProba: true, decisionFunction: false, sampleWeight: false, csr: false, earlyStopping: false }
    },
    get isFitted() { return fitted },
    get isDisposed() { return disposed }
  }
}

// Mock estimator without predictProba (last step)
function createMockRegressor() {
  let fitted = false
  let disposed = false
  return {
    fit(X, y) { fitted = true; return this },
    predict(X) {
      return new Float64Array(X.rows)
    },
    score(X, y) { return 0.8 },
    save() {
      return encodeBundle(
        { typeId: 'wlearn.mock.regressor@1' },
        [{ id: 'state', data: new Uint8Array([3]) }]
      )
    },
    dispose() { disposed = true },
    getParams() { return { type: 'regressor' } },
    setParams(p) { return this },
    get capabilities() {
      return { classifier: false, regressor: true, predictProba: false, decisionFunction: false, sampleWeight: false, csr: false, earlyStopping: false }
    },
    get isFitted() { return fitted },
    get isDisposed() { return disposed }
  }
}

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6]), rows: 3, cols: 2 }
const y = new Float64Array([0, 1, 0])

describe('Pipeline', () => {
  it('requires at least one step', () => {
    assert.throws(() => new Pipeline([]), ValidationError)
  })

  it('fit transforms through chain and fits last', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])

    assert.equal(pipe.isFitted, false)
    pipe.fit(X, y)
    assert.equal(pipe.isFitted, true)
    assert.equal(t1.isFitted, true)
    assert.equal(clf.isFitted, true)
  })

  it('predict transforms then predicts', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    const preds = pipe.predict(X)
    assert(preds instanceof Float64Array)
    assert.equal(preds.length, 3)
  })

  it('predictProba works with classifier', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    const proba = pipe.predictProba(X)
    assert(proba instanceof Float64Array)
    assert.equal(proba.length, 6) // 3 rows * 2 classes
  })

  it('predictProba throws if last step lacks it', () => {
    const t1 = createMockTransformer('t1')
    const reg = createMockRegressor()
    const pipe = new Pipeline([['transform', t1], ['regress', reg]])
    pipe.fit(X, y)

    assert.throws(() => pipe.predictProba(X), ValidationError)
  })

  it('score transforms then scores', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    const s = pipe.score(X, y)
    assert.equal(s, 0.9)
  })

  it('capabilities reflects last step', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])

    assert.equal(pipe.capabilities.classifier, true)
    assert.equal(pipe.capabilities.predictProba, true)
  })

  it('getParams returns per-step params', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])

    const params = pipe.getParams()
    assert.deepEqual(params.transform, { name: 't1' })
    assert.deepEqual(params.classify, { type: 'classifier' })
  })

  it('throws NotFittedError before fit', () => {
    const clf = createMockClassifier()
    const pipe = new Pipeline([['classify', clf]])
    assert.throws(() => pipe.predict(X), NotFittedError)
    assert.throws(() => pipe.score(X, y), NotFittedError)
  })

  it('save produces valid WLRN bundle', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    const bytes = pipe.save()
    assert(bytes instanceof Uint8Array)

    // Verify it's a valid bundle
    const { manifest, toc, blobs } = decodeBundle(bytes)
    assert.equal(manifest.typeId, 'wlearn.pipeline@1')
    assert.equal(manifest.steps.length, 2)
    assert.equal(manifest.steps[0].name, 'transform')
    assert.equal(manifest.steps[1].name, 'classify')
    assert.equal(toc.length, 2)

    // Each step blob should be a valid WLRN bundle
    for (const entry of toc) {
      const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
      const inner = decodeBundle(blob)
      assert(inner.manifest.typeId)
    }
  })

  it('save throws if not fitted', () => {
    const clf = createMockClassifier()
    const pipe = new Pipeline([['classify', clf]])
    assert.throws(() => pipe.save(), NotFittedError)
  })
})

describe('Pipeline.load', () => {
  it('round-trips via save/load', async () => {
    // Register mock loaders
    register('wlearn.mock.transformer.t1@1', (manifest, toc, blobs) => {
      const t = createMockTransformer('t1')
      t.fit({ data: new Float64Array(1), rows: 1, cols: 1 }, new Float64Array(1))
      return t
    })
    register('wlearn.mock.classifier@1', (manifest, toc, blobs) => {
      const c = createMockClassifier()
      c.fit({ data: new Float64Array(1), rows: 1, cols: 1 }, new Float64Array(1))
      return c
    })

    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    const bytes = pipe.save()
    const loaded = await Pipeline.load(bytes)

    assert.equal(loaded.isFitted, true)
    const preds = loaded.predict(X)
    assert(preds instanceof Float64Array)
    assert.equal(preds.length, 3)

    loaded.dispose()
  })
})

describe('Pipeline dispose', () => {
  it('disposes all steps', () => {
    const t1 = createMockTransformer('t1')
    const clf = createMockClassifier()
    const pipe = new Pipeline([['transform', t1], ['classify', clf]])
    pipe.fit(X, y)

    pipe.dispose()
    assert.equal(t1.isDisposed, true)
    assert.equal(clf.isDisposed, true)
  })

  it('double-dispose does not crash', () => {
    const clf = createMockClassifier()
    const pipe = new Pipeline([['classify', clf]])
    pipe.fit(X, y)
    pipe.dispose()
    pipe.dispose() // should not throw
  })

  it('use-after-dispose throws DisposedError', () => {
    const clf = createMockClassifier()
    const pipe = new Pipeline([['classify', clf]])
    pipe.fit(X, y)
    pipe.dispose()

    assert.throws(() => pipe.predict(X), DisposedError)
    assert.throws(() => pipe.fit(X, y), DisposedError)
    assert.throws(() => pipe.score(X, y), DisposedError)
    assert.throws(() => pipe.setParams({}), DisposedError)
  })
})
