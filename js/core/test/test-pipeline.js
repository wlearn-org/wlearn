const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { Pipeline } = require('../src/pipeline.js')
const { register } = require('../src/registry.js')
const { encodeBundle, decodeBundle } = require('../src/bundle.js')
const { DisposedError, NotFittedError, ValidationError } = require('../src/errors.js')

it('rejects a transductive intermediate before fitting any step', () => {
  let calls = 0
  const first = { transform: X => X, fitTransform(X) { calls++; return X } }
  const embedding = { capabilities: { transductive: true }, fitTransform(X) { calls++; return X } }
  const final = { fit() { calls++ } }
  const pipe = new Pipeline([['first', first], ['embedding', embedding], ['final', final]])
  assert.throws(() => pipe.fit([[1]], [0]), /embedding.*transform/)
  assert.equal(calls, 0)
})

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
    get classes() { return new Int32Array([1, 0]) },
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

  it('Promise-lifts intermediate and final fit without an eager fitted state', async () => {
    const transformer = createMockTransformer('async')
    const classifier = createMockClassifier()
    let fittedInput = null
    transformer.fitTransform = async function (input, labels) {
      await Promise.resolve()
      this.fit(input, labels)
      return this.transform(input)
    }
    classifier.fit = async function (input) {
      await Promise.resolve()
      fittedInput = input
      return this
    }
    const pipe = new Pipeline([
      ['transform', transformer], ['classify', classifier]
    ])
    const pending = pipe.fit(X, y)
    assert(pending instanceof Promise)
    assert.equal(pipe.isFitted, false)
    assert.throws(() => pipe.predict(X), NotFittedError)
    assert.strictEqual(await pending, pipe)
    assert.equal(pipe.isFitted, true)
    assert.equal(fittedInput.data[0], X.data[0] * 2)
    pipe.dispose()
  })

  it('keeps rejected asynchronous fits unfitted and guards pending disposal', async () => {
    let settle = null
    let rejectFit = true
    const classifier = createMockClassifier()
    classifier.fit = function () {
      return new Promise((resolve, reject) => {
        settle = () => rejectFit
          ? reject(new Error('async fit failed'))
          : resolve(this)
      })
    }
    const pipe = new Pipeline([['classify', classifier]])
    const rejected = pipe.fit(X, y)
    assert.throws(() => pipe.fit(X, y), /already in progress/)
    assert.throws(
      () => pipe.setParams({ classify: {} }), /while fit is in progress/
    )
    settle()
    await assert.rejects(rejected, /async fit failed/)
    assert.equal(pipe.isFitted, false)

    rejectFit = false
    const pending = pipe.fit(X, y)
    assert.throws(() => pipe.dispose(), /while fit is in progress/)
    settle()
    await pending
    assert.equal(pipe.isFitted, true)
    pipe.dispose()
    assert.equal(pipe.isFitted, false)
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
    assert.deepEqual([...pipe.classes], [1, 0])
  })

  it('does not report fitted after disposal', () => {
    const pipe = new Pipeline([['classify', createMockClassifier()]])
    pipe.fit(X, y)
    pipe.dispose()
    assert.equal(pipe.isFitted, false)
    assert.throws(() => pipe.classes, DisposedError)
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
    const snapshot = pipe.capabilities
    snapshot.predictProba = false
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

  it('setParams invalidates fit before child mutation', () => {
    const transformer = createMockTransformer('transform')
    const classifier = createMockClassifier()
    const pipe = new Pipeline([
      ['transform', transformer], ['classify', classifier]
    ])
    pipe.fit(X, y)
    pipe.setParams({ transform: { changed: true } })
    assert.equal(pipe.isFitted, false)
    assert.throws(() => pipe.predict(X), NotFittedError)

    pipe.fit(X, y)
    transformer.setParams = () => { throw new Error('mutation failed') }
    assert.throws(
      () => pipe.setParams({ transform: { changed: true } }),
      /mutation failed/
    )
    assert.equal(pipe.isFitted, false)
    pipe.dispose()
  })

  it('rejects unknown steps and preflights every selected setter', () => {
    const transformer = createMockTransformer('transform')
    const classifier = createMockClassifier()
    let transformerMutations = 0
    transformer.setParams = () => { transformerMutations++; return transformer }
    classifier.setParams = undefined
    const pipe = new Pipeline([
      ['transform', transformer], ['classify', classifier]
    ])
    pipe.fit(X, y)

    assert.throws(
      () => pipe.setParams({ clasify: {} }),
      error => error instanceof ValidationError && /Unknown.*clasify/.test(error.message)
    )
    assert.equal(pipe.isFitted, true)
    assert.throws(
      () => pipe.setParams({ transform: {}, classify: {} }),
      error => error instanceof ValidationError && /classify.*setParams/.test(error.message)
    )
    assert.equal(transformerMutations, 0)
    assert.equal(pipe.isFitted, true)
    pipe.dispose()
  })

  it('defensively snapshots candidate provenance', () => {
    const source = {
      candidate: {
        model: { classId: 'wlearn.test.model@1', params: { depth: 3 } },
      },
    }
    const pipe = new Pipeline(
      [['classify', createMockClassifier()]],
      { provenance: source }
    )
    source.candidate.model.params.depth = 99
    assert.equal(pipe.provenance.candidate.model.params.depth, 3)
    const exposed = pipe.provenance
    exposed.candidate.model.params.depth = 7
    assert.equal(pipe.provenance.candidate.model.params.depth, 3)
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
    const provenance = {
      candidate: {
        model: { classId: 'wlearn.test.model@1', params: { depth: 3 } },
      },
    }
    const pipe = new Pipeline(
      [['transform', t1], ['classify', clf]],
      { provenance }
    )
    pipe.fit(X, y)

    const bytes = pipe.save()
    assert(bytes instanceof Uint8Array)

    // Verify it's a valid bundle
    const { manifest, toc, blobs } = decodeBundle(bytes)
    assert.equal(manifest.typeId, 'wlearn.pipeline@1')
    assert.equal(manifest.steps.length, 2)
    assert.equal(manifest.steps[0].name, 'transform')
    assert.equal(manifest.steps[1].name, 'classify')
    assert.deepEqual(manifest.metadata.provenance, provenance)
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
    const provenance = {
      candidate: {
        model: { classId: 'wlearn.test.model@1', params: { depth: 3 } },
      },
    }
    const pipe = new Pipeline(
      [['transform', t1], ['classify', clf]],
      { provenance }
    )
    pipe.fit(X, y)

    const bytes = pipe.save()
    const loaded = await Pipeline.load(bytes)

    assert.equal(loaded.isFitted, true)
    const preds = loaded.predict(X)
    assert(preds instanceof Float64Array)
    assert.equal(preds.length, 3)
    assert.deepEqual(loaded.provenance, provenance)

    loaded.dispose()
  })

  it('forwards one load context to every nested step', async () => {
    const contexts = []
    register('wlearn.mock.transformer.context@1', (manifest, toc, blobs, context) => {
      contexts.push(context)
      const transformer = createMockTransformer('context')
      transformer.fit(X, y)
      return transformer
    }, { acceptsContext: true })
    register('wlearn.mock.classifier@1', (manifest, toc, blobs, context) => {
      contexts.push(context)
      const classifier = createMockClassifier()
      classifier.fit(X, y)
      return classifier
    }, { acceptsContext: true })

    const pipeline = new Pipeline([
      ['transform', createMockTransformer('context')],
      ['classify', createMockClassifier()]
    ])
    pipeline.fit(X, y)
    const runtimeOptions = { maxPlanBytes: 123 }
    const loaded = await Pipeline.load(pipeline.save(), {
      loaderOptions: { 'wlearn.preprocess.tabular@1': runtimeOptions }
    })

    assert.equal(contexts.length, 2)
    assert.strictEqual(contexts[0], contexts[1])
    assert.deepEqual(
      contexts[0].loaderOptions['wlearn.preprocess.tabular@1'], runtimeOptions
    )
    assert.notStrictEqual(
      contexts[0].loaderOptions['wlearn.preprocess.tabular@1'], runtimeOptions
    )
    loaded.dispose()
    pipeline.dispose()
  })

  it('verifies the outer bundle hash before loading steps', async () => {
    const pipe = new Pipeline([['classify', createMockClassifier()]])
    pipe.fit(X, y)
    const bytes = pipe.save()
    const { toc } = decodeBundle(bytes)
    const originalHash = new TextEncoder().encode(toc[0].sha256)
    const replacement = new TextEncoder().encode('0'.repeat(64))
    const corrupted = bytes.slice()
    let replacements = 0
    for (let i = 0; i <= corrupted.length - originalHash.length; i++) {
      let matches = true
      for (let j = 0; j < originalHash.length; j++) {
        if (corrupted[i + j] !== originalHash[j]) { matches = false; break }
      }
      if (!matches) continue
      corrupted.set(replacement, i)
      replacements++
      i += originalHash.length - 1
    }
    assert.equal(replacements, 2, 'manifest and TOC hash declarations should both change')
    await assert.rejects(() => Pipeline.load(corrupted), /SHA-256 mismatch/)
    pipe.dispose()
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

describe('Pipeline persistence contracts', () => {
  for (const method of ['save', 'getParams']) it(`reports a step missing ${method} before serializing children`, () => {
    const first = createMockTransformer('first')
    let saves = 0
    first.save = () => { saves++; return new Uint8Array() }
    const last = createMockRegressor()
    last[method] = undefined
    const pipe = new Pipeline([['first', first], ['broken', last]])
    try {
      pipe.fit({ data: Float64Array.of(0, 1), rows: 2, cols: 1 }, [0, 1])
      assert.throws(() => pipe.save(), e => e instanceof ValidationError && e.message.includes('broken') && e.message.includes(method))
      assert.equal(saves, 0)
    } finally { pipe.dispose() }
  })
})

describe('Pipeline sample weights', () => {
  it('routes weights to supporting steps and rejects unsupported final estimators before mutation', () => {
    const weights = Float64Array.of(1, 3), calls = []
    const map = { capabilities: { transformer: true, sampleWeight: true },
      fitTransform(X, y, opts) { calls.push(['map', opts?.sampleWeight]); return X } }
    const final = { capabilities: { sampleWeight: true }, fit(X, y, opts) { calls.push(['model', opts?.sampleWeight]); return this } }
    const pipe = new Pipeline([['map', map], ['model', final]])
    pipe.fit([[0], [1]], [0, 1], { sampleWeight: weights })
    assert.deepEqual(calls, [['map', weights], ['model', weights]])
    final.capabilities.sampleWeight = false
    assert.throws(() => pipe.fit([[0], [1]], [0, 1], { sampleWeight: weights }), ValidationError)
    assert.equal(calls.length, 2)
  })
  it('fits unweighted transformers normally and validates weight shape and values first', () => {
    const calls = []
    const prep = { fitTransform(...args) { calls.push(args.length); return args[0] } }
    const final = { capabilities: { sampleWeight: true }, fit() { calls.push('fit'); return this } }
    const pipe = new Pipeline([['prep', prep], ['model', final]])
    for (const weight of [[1], [0, 0], [-1, 2], [NaN, 1], [Infinity, 1]]) {
      assert.throws(() => pipe.fit([[0], [1]], [0, 1], { sampleWeight: weight }), ValidationError)
    }
    assert.deepEqual(calls, [])
    pipe.fit([[0], [1]], [0, 1], { sampleWeight: [1, 2] })
    assert.deepEqual(calls, [2, 'fit'])
  })
})

it('Pipeline lifts async transforms during inference and forwards prediction options', async () => {
  const X = { rows: 2, cols: 1, data: Float64Array.of(1, 2) }
  const transform = { fit() {}, async transform(X) { return { ...X, data: Float64Array.from(X.data, v => 2 * v) } } }
  const model = {
    fit() {}, predict(X, opts) { assert.equal(opts.offset, 3); return Float64Array.from(X.data, v => v + opts.offset) }
  }
  const pipe = new Pipeline([['t', transform], ['m', model]])
  await pipe.fit(X, [1, 2])
  assert.deepEqual(Array.from(await pipe.predict(X, { offset: 3 })), [5, 7])
})

it('Pipeline routes weights by target rows for matrix targets', () => {
  const X = { rows: 2, cols: 1, data: Float64Array.of(1, 2) }
  const Y = { rows: 2, cols: 2, data: Float64Array.of(1, 2, 3, 4) }
  let observed
  const model = { capabilities: { sampleWeight: true }, fit(X, y, opts) { observed = opts.sampleWeight } }
  new Pipeline([['m', model]]).fit(X, Y, { sampleWeight: [1, 2] })
  assert.deepEqual(observed, [1, 2])
})
