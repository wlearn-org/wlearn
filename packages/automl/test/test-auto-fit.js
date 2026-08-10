const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { autoFit, registerBayesianSearch } = require('../src/auto-fit.js')
const { SearchableMock, SearchableMockReg, MockModel } = require('./mock-model.js')
const {
  ValidationError, BackendError, Pipeline, decodeBundle, load,
} = require('@wlearn/core')
const { Preprocessor } = require('@wlearn/preprocess')
const { VotingEnsemble } = require('@wlearn/ensemble')
const {
  createCandidate, makeCandidateId, seedFor,
} = require('../src/candidate.js')

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1.1, 2.3, 3.7, 4.2, 5.8, 6.1, 7.5, 8.9, 9.4, 10.6])

function expectedProvenance(candidate, baseSeed = 42, foldCount = 2) {
  return {
    candidateId: makeCandidateId(candidate),
    candidate,
    baseSeed,
    foldSeeds: Array.from({ length: foldCount }, (_unused, foldId) => ({
      foldId,
      seed: seedFor(candidate, foldId, baseSeed),
    })),
  }
}

describe('autoFit classification', () => {
  it('returns fitted model with refit=true', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 3, cv: 2, task: 'classification' }
    )
    assert(result.model !== null)
    assert(result.model.isFitted)
    const preds = result.model.predict(X)
    assert.equal(preds.length, 10)
    result.model.dispose()
  })

  it('returns leaderboard', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 5, cv: 2, task: 'classification' }
    )
    assert(result.leaderboard.length === 5)
    assert.equal(result.archive.size, result.leaderboard.length)
    assert.equal(result.archive.records({ status: 'ok' }).length, result.leaderboard.length)
    assert(result.bestScore >= 0)
    assert.equal(result.bestModelName, 'mock')
    if (result.model) result.model.dispose()
  })

  it('accepts EstimatorSpec tuples', async () => {
    const result = await autoFit(
      [['mock', SearchableMock, { task: 'classification' }]],
      X, yCls,
      { nIter: 2, cv: 2, task: 'classification' }
    )
    assert(result.model !== null)
    assert(result.leaderboard.length === 2)
    if (result.model) result.model.dispose()
  })

  it('refit=false and ensemble=false returns null model', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 2, cv: 2, task: 'classification', refit: false, ensemble: false }
    )
    assert.equal(result.model, null)
    assert(result.bestScore >= 0)
  })
})

describe('autoFit ensemble', () => {
  it('ensemble=true returns VotingEnsemble', async () => {
    const result = await autoFit(
      [
        { name: 'm1', classId: 'wlearn.test.m1@1', cls: SearchableMock },
        { name: 'm2', classId: 'wlearn.test.m2@1', cls: SearchableMock },
      ],
      X, yCls,
      { nIter: 3, cv: 2, task: 'classification', ensemble: true, ensembleSize: 5 }
    )
    assert(result.model !== null)
    // VotingEnsemble has capabilities property
    assert(result.model.capabilities.classifier)
    const preds = result.model.predict(X)
    assert.equal(preds.length, 10)
    result.model.dispose()
  })

  it('disposes a final ensemble whose fit fails and preserves the fit error', async () => {
    const originalCreate = VotingEnsemble.create
    const fitError = new Error('ensemble fit failed')
    let disposals = 0
    VotingEnsemble.create = async () => ({
      async fit() { throw fitError },
      dispose() {
        disposals++
        throw new Error('ensemble cleanup failed')
      },
    })
    try {
      await assert.rejects(
        () => autoFit(
          [{ name: 'mock', cls: SearchableMock }], X, yCls,
          { nIter: 1, cv: 2, task: 'classification', ensemble: true }
        ),
        error => error === fitError
      )
    } finally {
      VotingEnsemble.create = originalCreate
    }
    assert.equal(disposals, 1)
  })
})

describe('autoFit regression', () => {
  it('works with regression task', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMockReg }],
      X, yReg,
      { nIter: 3, cv: 2, task: 'regression' }
    )
    assert(result.model !== null)
    assert(isFinite(result.bestScore))
    result.model.dispose()
  })
})

describe('autoFit preprocessing isolation', () => {
  const templates = [
    {
      templateId: 'plain',
      typeId: 'wlearn.preprocess.tabular@1',
      params: {
        encode: false, impute: false, scale: false, maxCategories: 2,
      },
    },
    {
      templateId: 'scaled',
      typeId: 'wlearn.preprocess.tabular@1',
      params: {
        encode: false, impute: false, scale: 'standard', maxCategories: 2,
      },
    },
  ]

  it('rejects explicit non-object template params and search spaces', async () => {
    const base = {
      templateId: 'invalid', typeId: 'wlearn.preprocess.tabular@1'
    }
    for (const params of [false, 0, '', [], null]) {
      await assert.rejects(
        () => autoFit(
          [{ name: 'mock', cls: SearchableMock }], X, yCls,
          { preprocess: [{ ...base, params }], ensemble: false, refit: false }
        ),
        error => error instanceof ValidationError && /params must be an object/.test(error.message)
      )
    }
    for (const searchSpace of [false, 0, '', [], null]) {
      await assert.rejects(
        () => autoFit(
          [{ name: 'mock', cls: SearchableMock }], X, yCls,
          { preprocess: [{ ...base, searchSpace }], ensemble: false, refit: false }
        ),
        error => error instanceof ValidationError && /searchSpace must be an object/.test(error.message)
      )
    }
  })

  it('preserves preprocessing backend initialization errors before model creation', async () => {
    const originalCreate = Preprocessor.create
    let modelCreates = 0
    class CountingModel extends SearchableMock {
      static async create(params) {
        modelCreates++
        return super.create(params)
      }
    }
    try {
      for (const backend of ['native', 'wasm']) {
        const backendError = new BackendError(`${backend} Tranfi unavailable`)
        Preprocessor.create = async () => { throw backendError }
        await assert.rejects(
          () => autoFit(
            [{ name: 'mock', cls: CountingModel }], X, yCls,
            {
              nIter: 1, cv: 2, task: 'classification',
              preprocess: true, ensemble: false, refit: false,
            }
          ),
          error => error === backendError
        )
      }
    } finally {
      Preprocessor.create = originalCreate
    }
    assert.equal(modelCreates, 0)
  })

  it('evaluates every resolved preprocessing template', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X,
      yCls,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: templates,
        ensemble: false,
        refit: false,
      }
    )

    assert.deepEqual(
      result.leaderboard.map(
        entry => entry.candidate.preprocess.templateId
      ).sort(),
      ['plain', 'scaled']
    )
    assert.equal(new Set(
      result.leaderboard.map(entry => entry.candidateId)
    ).size, 2)
    assert.equal(result.bestParams.preprocess.policyVersion, 1)
  })

  it('fits categorical encoding inside each CV fold', async () => {
    const trainCols = []
    const predictCols = []

    class ShapeProbe {
      static get classId() { return 'wlearn.test.shape-probe@1' }

      static defaultSearchSpace() {
        return { task: { type: 'categorical', values: ['classification'] } }
      }

      static async create(params = {}) {
        return new ShapeProbe(params)
      }

      constructor(params) {
        this.params = { ...params }
        this.fitted = false
      }

      fit(X) {
        trainCols.push(X.cols)
        this.fitted = true
        return this
      }

      predict(X) {
        predictCols.push(X.cols)
        return new Int32Array(X.rows)
      }

      score() { return 0 }
      getParams() { return { ...this.params } }
      setParams(p) { Object.assign(this.params, p); return this }
      dispose() { this.fitted = false }
      get isFitted() { return this.fitted }
      get capabilities() { return { classifier: true, regressor: false } }
    }

    const categoricalX = {
      data: new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]),
      rows: 8,
      cols: 1,
    }
    const categoricalY = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1])

    await autoFit(
      [{ name: 'shape', cls: ShapeProbe }],
      categoricalX,
      categoricalY,
      {
        nIter: 1,
        cv: 2,
        seed: 42,
        task: 'classification',
        preprocess: { encode: 'onehot', impute: false, scale: false },
        ensemble: false,
        refit: false,
      }
    )

    // Each training fold sees four unique categories. A leaked full-data fit
    // would expose all eight columns to both fold models.
    assert.deepEqual(trainCols, [4, 4])
    assert.deepEqual(predictCols, [4, 4])
  })

  it('fits scaling statistics on each training fold only', async () => {
    const trainMeans = []

    class MeanProbe {
      static get classId() { return 'wlearn.test.mean-probe@1' }

      static defaultSearchSpace() {
        return { task: { type: 'categorical', values: ['regression'] } }
      }
      static async create(params = {}) { return new MeanProbe(params) }
      constructor(params) { this.params = { ...params }; this.fitted = false }
      fit(X) {
        let sum = 0
        for (const value of X.data) sum += value
        trainMeans.push(sum / X.data.length)
        this.fitted = true
        return this
      }
      predict(X) { return new Float64Array(X.rows) }
      score() { return 0 }
      getParams() { return { ...this.params } }
      setParams(p) { Object.assign(this.params, p); return this }
      dispose() { this.fitted = false }
      get isFitted() { return this.fitted }
      get capabilities() { return { classifier: false, regressor: true } }
    }

    const numericX = {
      data: new Float64Array([0.1, 0.2, 0.3, 0.4, 10.1, 20.2, 30.3, 1000.4]),
      rows: 8,
      cols: 1,
    }

    await autoFit(
      [{ name: 'mean', cls: MeanProbe }],
      numericX,
      new Float64Array([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4]),
      {
        nIter: 1,
        cv: 2,
        seed: 7,
        task: 'regression',
        preprocess: { encode: false, impute: false, scale: 'standard' },
        ensemble: false,
        refit: false,
      }
    )

    assert.equal(trainMeans.length, 2)
    for (const mean of trainMeans) assert(Math.abs(mean) < 1e-12)
  })

  it('returns a fitted Pipeline when preprocessing and refit are enabled', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X,
      yCls,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: { encode: false, scale: 'standard', maxCategories: 2 },
        ensemble: false,
        refit: true,
      }
    )

    assert(result.model instanceof Pipeline)
    assert.equal(result.preprocessor, null)
    assert.equal(result.model.predict(X).length, X.rows)
    assert.deepEqual(Object.keys(result.model.getParams()), ['preprocess', 'model'])
    result.model.dispose()
  })

  it('round-trips candidate provenance in a fitted Pipeline', async () => {
    const adversarialParams = JSON.parse('{"__proto__":{"safe":true}}')
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock, params: adversarialParams }],
      X,
      yCls,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: templates[1].params,
        ensemble: false,
        refit: true,
      }
    )
    const expected = expectedProvenance(result.bestCandidate)
    assert(Object.hasOwn(expected.candidate.model.params, '__proto__'))
    assert.deepEqual(result.model.provenance, expected)

    const bytes = result.model.save()
    const { manifest } = decodeBundle(bytes)
    assert.deepEqual(manifest.metadata.provenance, expected)
    const loaded = await load(bytes)
    assert.deepEqual(loaded.provenance, expected)
    assert.deepEqual([...loaded.predict(X)], [...result.model.predict(X)])

    loaded.dispose()
    result.model.dispose()
  })

  it('round-trips candidate Pipelines nested in an ensemble', async () => {
    const result = await autoFit(
      [
        {
          name: 'm1', classId: 'wlearn.test.persist.m1@1',
          cls: SearchableMock,
          params: JSON.parse('{"__proto__":{"model":"m1"}}'),
        },
        {
          name: 'm2', classId: 'wlearn.test.persist.m2@1',
          cls: SearchableMock,
          params: JSON.parse('{"__proto__":{"model":"m2"}}'),
        },
      ],
      X,
      yCls,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: templates[0].params,
        ensemble: true,
        ensembleSize: 2,
      }
    )
    const bytes = result.model.save()
    const { toc, blobs } = decodeBundle(bytes)
    assert(toc.length > 0)
    for (const entry of toc) {
      const child = decodeBundle(
        blobs.subarray(entry.offset, entry.offset + entry.length)
      )
      assert.equal(child.manifest.typeId, 'wlearn.pipeline@1')
      assert.equal(
        child.manifest.metadata.provenance.candidate.preprocess.templateId,
        'wlearn.preprocess.inline.v1'
      )
      assert(Object.hasOwn(
        child.manifest.metadata.provenance.candidate.model.params,
        '__proto__'
      ))
      assert.equal(
        child.manifest.metadata.provenance.candidateId,
        makeCandidateId(child.manifest.metadata.provenance.candidate)
      )
    }

    const loaded = await load(bytes)
    assert.deepEqual([...loaded.predict(X)], [...result.model.predict(X)])
    loaded.dispose()
    result.model.dispose()
  })

  it('rejects raw-feature stacking passthrough with preprocessing', async () => {
    await assert.rejects(() => autoFit(
      [
        {
          name: 'm1', classId: 'wlearn.test.stack.m1@1',
          cls: SearchableMock,
        },
        {
          name: 'm2', classId: 'wlearn.test.stack.m2@1',
          cls: SearchableMock,
        },
      ],
      X,
      yCls,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: templates[0].params,
        ensemble: true,
        ensembleSize: 2,
        stacking: true,
        metaEstimator: { cls: SearchableMock, params: {} },
        stackingPassthrough: true,
      }
    ), /stackingPassthrough=true/)
  })
})

describe('autoFit onProgress', () => {
  it('calls onProgress for each candidate during search', async () => {
    const events = []
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      {
        nIter: 3, cv: 2, task: 'classification',
        ensemble: false, refit: true,
        onProgress: (e) => events.push(e),
      }
    )
    assert.equal(events.length, 3)
    for (const e of events) {
      assert.equal(e.phase, 'search')
      assert.equal(typeof e.candidatesDone, 'number')
      assert.equal(typeof e.bestScore, 'number')
      assert.equal(typeof e.bestModel, 'string')
      assert.equal(typeof e.lastCandidate.model, 'string')
      assert.equal(typeof e.lastCandidate.score, 'number')
      assert.equal(typeof e.lastCandidate.timeMs, 'number')
      assert.equal(typeof e.elapsedMs, 'number')
    }
    assert.equal(events[0].candidatesDone, 1)
    assert.equal(events[2].candidatesDone, 3)
    if (result.model) result.model.dispose()
  })

  it('emits ensemble phase event when ensemble=true', async () => {
    const events = []
    const result = await autoFit(
      [
        { name: 'm1', classId: 'wlearn.test.m1@1', cls: SearchableMock },
        { name: 'm2', classId: 'wlearn.test.m2@1', cls: SearchableMock },
      ],
      X, yCls,
      {
        nIter: 2, cv: 2, task: 'classification',
        ensemble: true, ensembleSize: 3,
        onProgress: (e) => events.push(e),
      }
    )
    const phases = events.map(e => e.phase)
    assert(phases.includes('search'))
    assert(phases.includes('ensemble'))
    if (result.model) result.model.dispose()
  })

  it('works with portfolio strategy', async () => {
    const events = []
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      {
        strategy: 'portfolio', cv: 2, task: 'classification',
        ensemble: false, refit: true,
        onProgress: (e) => events.push(e),
      }
    )
    assert(events.length > 0)
    assert.equal(events[0].phase, 'search')
    if (result.model) result.model.dispose()
  })
})

describe('autoFit bayesian backend', () => {
  it('uses the built-in BayesianSearch by default', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      {
        strategy: 'bayesian',
        task: 'classification',
        nIter: 3,
        nInitial: 1,
        cv: 2,
        ensemble: false,
        refit: false,
      }
    )
    assert.equal(result.model, null)
    assert.equal(result.leaderboard.length, 3)
    assert.equal(result.archive.size, 3)
    assert.equal(result.bestModelName, 'mock')
  })

  it('uses a registered BayesianSearch constructor', async () => {
    let constructed = false
    class FakeBayesianSearch {
      constructor(specs, opts) {
        constructed = true
        this.specs = specs
        this.opts = opts
      }

      async fit() {
        const candidate = createCandidate({
          displayName: 'mock',
          classId: SearchableMock.classId
        }, {})
        return {
          leaderboard: {
            ranked() {
              return [{
                id: 'fake-1', candidateId: makeCandidateId(candidate),
                candidate, modelName: 'mock', params: {}, meanScore: 1
              }]
            }
          },
          archive: { size: 1 },
          bestResult: {
            candidateId: makeCandidateId(candidate), candidate,
            params: {}, modelName: 'mock', meanScore: 1
          },
        }
      }
    }

    try {
      registerBayesianSearch(FakeBayesianSearch)
      const result = await autoFit(
        [{ name: 'mock', cls: SearchableMock }],
        X, yCls,
        { strategy: 'bayesian', task: 'classification', ensemble: false, refit: false }
      )
      assert.equal(constructed, true)
      assert.equal(result.bestModelName, 'mock')
      assert.equal(result.bestScore, 1)
    } finally {
      registerBayesianSearch(null)
    }
  })

  it('can reset a registered BayesianSearch constructor', async () => {
    class FakeBayesianSearch {}
    registerBayesianSearch(FakeBayesianSearch)
    assert.doesNotThrow(() => registerBayesianSearch(null))
  })
})

describe('autoFit validation', () => {
  it('throws on empty models', async () => {
    await assert.rejects(
      () => autoFit([], X, yCls, { task: 'classification' }),
      ValidationError
    )
  })

  it('auto-detects classification task from Int32Array', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 2, cv: 2 }
    )
    assert(result.bestScore >= 0 && result.bestScore <= 1)
    if (result.model) result.model.dispose()
  })

  it('auto-detects regression task from Float64Array', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMockReg }],
      X, yReg,
      { nIter: 2, cv: 2 }
    )
    assert(isFinite(result.bestScore))
    if (result.model) result.model.dispose()
  })
})
