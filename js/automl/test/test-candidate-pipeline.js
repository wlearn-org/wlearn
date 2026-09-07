'use strict'

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { Pipeline, BackendError } = require('@wlearn/core')
const {
  createCandidate, makeCandidateId,
} = require('../src/candidate.js')
const {
  createCandidatePipelineClass, fitCandidate,
} = require('../src/candidate-pipeline.js')
const { Executor } = require('../src/executor.js')
const { RandomSearch } = require('../src/search.js')
const { SuccessiveHalvingSearch } = require('../src/halving.js')
const { ProgressiveSearch } = require('../src/progressive.js')
const { PortfolioSearch } = require('../src/portfolio.js')
const { BayesianSearch } = require('../src/bayesian.js')

const SEARCH_FACTORIES = [
  ['random', spec => new RandomSearch([spec], {
    nIter: 1, cv: 2, task: 'classification', seed: 7,
  })],
  ['halving', spec => new SuccessiveHalvingSearch([spec], {
    nIter: 1, cv: 2, task: 'classification', seed: 7, factor: 2,
  })],
  ['progressive', spec => new ProgressiveSearch([spec], {
    nIter: 1, cv: 2, task: 'classification', seed: 7,
    promoteCount: 1, probeFraction: 1,
  })],
  ['portfolio', spec => new PortfolioSearch([spec], {
    cv: 2, task: 'classification', seed: 7,
  })],
  ['bayesian', spec => new BayesianSearch([spec], {
    nIter: 1, nInitial: 0, cv: 2, task: 'classification', seed: 7,
  })],
]

function candidateFor(classId = 'wlearn.test.lifecycle@1') {
  return createCandidate({
    displayName: 'lifecycle',
    classId,
  }, {}, {
    templateId: 'plain',
    typeId: 'wlearn.preprocess.tabular@1',
    resolvedParams: { scale: false },
  })
}

function preprocessorClass(events, { disposeThrows = false } = {}) {
  return class FakePreprocessor {
    static async create() {
      events.push('preprocessor:create')
      return new FakePreprocessor()
    }

    fitTransform(X) { events.push('preprocessor:fit'); return X }
    transform(X) { events.push('preprocessor:transform'); return X }
    getParams() { return {} }
    dispose() {
      events.push('preprocessor:dispose')
      if (disposeThrows) throw new Error('preprocessor dispose failed')
    }
  }
}

function modelSpec(Model) {
  return {
    name: 'lifecycle',
    classId: 'wlearn.test.lifecycle@1',
    cls: Model,
  }
}

describe('preprocessed candidate ownership', () => {
  it('fitCandidate waits for asynchronous model fit', async () => {
    class DeferredFitModel {
      #fitted = false

      static async create() { return new DeferredFitModel() }
      async fit() {
        await Promise.resolve()
        this.#fitted = true
        return this
      }
      predict(input) {
        assert.equal(this.#fitted, true)
        return new Int32Array(input.rows)
      }
      dispose() {}
    }
    const candidate = createCandidate({
      displayName: 'deferred', classId: 'wlearn.test.deferred-fit@1',
    }, {})
    const fitted = await fitCandidate(
      { cls: DeferredFitModel }, candidate,
      { data: new Float64Array([1, 2]), rows: 2, cols: 1 },
      new Int32Array([0, 1]), makeCandidateId(candidate)
    )
    assert.deepEqual(fitted.predict({ rows: 2 }), new Int32Array(2))
    fitted.dispose()
  })

  it('releases the preprocessor when model construction fails', async () => {
    const events = []
    class FailingModel {
      static async create() {
        events.push('model:create')
        throw new Error('model create failed')
      }
    }
    const CandidateClass = createCandidatePipelineClass(
      modelSpec(FailingModel), candidateFor(),
      { Preprocessor: preprocessorClass(events), Pipeline }
    )

    await assert.rejects(() => CandidateClass.create(), /model create failed/)
    assert.deepEqual(events, [
      'preprocessor:create', 'model:create', 'preprocessor:dispose',
    ])
  })

  it('releases model then preprocessor when Pipeline construction fails', async () => {
    const events = []
    class Model {
      static async create() {
        events.push('model:create')
        return { dispose() { events.push('model:dispose') } }
      }
    }
    class FailingPipeline {
      constructor() {
        events.push('pipeline:create')
        throw new Error('pipeline create failed')
      }
    }
    const CandidateClass = createCandidatePipelineClass(
      modelSpec(Model), candidateFor(),
      { Preprocessor: preprocessorClass(events), Pipeline: FailingPipeline }
    )

    await assert.rejects(() => CandidateClass.create(), /pipeline create failed/)
    assert.deepEqual(events, [
      'preprocessor:create', 'model:create', 'pipeline:create',
      'model:dispose', 'preprocessor:dispose',
    ])
  })

  for (const phase of ['fit', 'predict']) {
    it(`Executor releases model then preprocessor after ${phase} failure`, async () => {
      const events = []
      class FailingModel {
        static async create() {
          events.push('model:create')
          return new FailingModel()
        }

        fit() {
          events.push('model:fit')
          if (phase === 'fit') throw new Error('fit failed')
          return this
        }

        predict() {
          events.push('model:predict')
          throw new Error('predict failed')
        }

        dispose() { events.push('model:dispose') }
        getParams() { return {} }
      }
      const candidate = candidateFor()
      const CandidateClass = createCandidatePipelineClass(
        modelSpec(FailingModel), candidate,
        { Preprocessor: preprocessorClass(events), Pipeline }
      )
      const executor = new Executor({
        folds: [{ train: new Int32Array([0]), test: new Int32Array([1]) }],
        scoring: 'accuracy',
        X: {
          data: new Float64Array([1, 2]), rows: 2, cols: 1,
        },
        y: new Int32Array([0, 1]),
      })

      await assert.rejects(() => executor.evaluateCandidate({
        candidateId: makeCandidateId(candidate),
        candidate,
        cls: CandidateClass,
        params: {},
      }), new RegExp(`${phase} failed`))
      assert.deepEqual(events.slice(-2), [
        'model:dispose', 'preprocessor:dispose',
      ])
    })
  }

  it('continues reverse cleanup after one child dispose throws', async () => {
    const events = []
    class Model {
      static async create() { return new Model() }
      fit() { return this }
      dispose() {
        events.push('model:dispose')
        throw new Error('model dispose failed')
      }
      getParams() { return {} }
    }
    const CandidateClass = createCandidatePipelineClass(
      modelSpec(Model), candidateFor(),
      { Preprocessor: preprocessorClass(events), Pipeline }
    )
    const pipeline = await CandidateClass.create()

    assert.throws(() => pipeline.dispose(), /model dispose failed/)
    assert.deepEqual(events.slice(-2), [
      'model:dispose', 'preprocessor:dispose',
    ])
  })

  for (const [name, makeSearch] of SEARCH_FACTORIES) {
    it(`${name} refit keeps its private winner and releases a failed Pipeline`, async () => {
      const events = []
      const fitError = new Error(`${name} refit failed`)
      let failFit = false
      class Model {
        static classId = `wlearn.test.${name}-refit@1`
        static defaultSearchSpace() { return {} }
        static async create(params = {}) {
          events.push(`model:create:${JSON.stringify(params)}`)
          return new Model()
        }
        fit() {
          events.push('model:fit')
          if (failFit) throw fitError
          return this
        }
        predict(X) { return new Int32Array(X.rows) }
        dispose() {
          events.push('model:dispose')
          if (failFit) throw new Error('model cleanup failed')
        }
        getParams() { return {} }
      }
      const preprocess = candidateFor(Model.classId).preprocess
      const spec = {
        name,
        classId: Model.classId,
        cls: Model,
        preprocessChoices: [preprocess],
      }
      spec.createCandidateClass = candidate => createCandidatePipelineClass(
        spec, candidate, {
          Preprocessor: preprocessorClass(events), Pipeline,
        }
      )
      const search = makeSearch(spec)
      const X = {
        data: new Float64Array([0, 1, 2, 3]), rows: 4, cols: 1,
      }
      const y = new Int32Array([0, 0, 1, 1])
      const result = await search.fit(X, y)
      const originalId = result.bestResult.candidateId
      const injected = createCandidate({
        displayName: name, classId: Model.classId,
      }, { unevaluated: true }, preprocess)
      const injectedId = makeCandidateId(injected)
      result.bestResult.candidate = injected
      result.bestResult.candidateId = injectedId
      const exposed = search.bestResult
      exposed.candidate = injected
      exposed.candidateId = injectedId
      result.leaderboard.add({
        candidateId: injectedId,
        candidate: injected,
        scores: new Float64Array([999]),
        fitTimeMs: 0,
      })

      assert.equal(search.bestResult.candidateId, originalId)
      events.length = 0
      failFit = true
      await assert.rejects(() => search.refitBest(X, y), error => error === fitError)
      assert.deepEqual(events, [
        'preprocessor:create', 'model:create:{"task":"classification"}',
        'preprocessor:fit', 'model:fit',
        'model:dispose', 'preprocessor:dispose',
      ])
    })
  }

  for (const [name, makeSearch] of SEARCH_FACTORIES) {
    it(`${name} preserves the first backend failure when no candidate succeeds`, async () => {
      const backendError = new BackendError(`${name} backend unavailable`)
      let modelCreates = 0
      class Model {
        static classId = `wlearn.test.${name}-backend-error@1`
        static defaultSearchSpace() { return {} }
        static async create() {
          modelCreates++
          throw new Error('base model must not be created')
        }
      }
      class FailingCandidate {
        static async create() { throw backendError }
      }
      const preprocess = candidateFor(Model.classId).preprocess
      const spec = {
        name,
        classId: Model.classId,
        cls: Model,
        preprocessChoices: [preprocess],
        createCandidateClass: () => FailingCandidate,
      }
      const search = makeSearch(spec)
      const X = {
        data: new Float64Array([0, 1, 2, 3]), rows: 4, cols: 1,
      }
      const y = new Int32Array([0, 0, 1, 1])
      await assert.rejects(() => search.fit(X, y), error => error === backendError)
      assert.equal(modelCreates, 0)
    })
  }
})
