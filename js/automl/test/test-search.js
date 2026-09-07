const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { RandomSearch } = require('../src/search.js')
const { SearchableMock, SearchableMockReg } = require('./mock-model.js')
const { ValidationError } = require('@wlearn/core')

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for (const [name, Search] of [
  ['random', RandomSearch],
  ['halving', require('../src/halving.js').SuccessiveHalvingSearch],
  ['progressive', require('../src/progressive.js').ProgressiveSearch],
  ['portfolio', require('../src/portfolio.js').PortfolioSearch],
]) {
  it(`${name} preserves explicit task and custom folds through search and refit`, async () => {
    const { createResamplingPlan } = require('@wlearn/core')
    const plan = createResamplingPlan({ strategy: 'group_kfold', n: 10, k: 2,
      groups: Int32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]) })
    const seen = []
    class Model {
      static classId = 'wlearn.test.task-folds@1'
      static async create(params) {
        assert.equal(params.task, 'regression')
        return new Model()
      }
      fit(Xtrain) { this.train = Array.from(Xtrain.data).filter((_, i) => i % 2 === 0); return this }
      predict(Xtest) {
        const test = Array.from(Xtest.data).filter((_, i) => i % 2 === 0)
        const group = v => Math.floor((v - 1) / 4)
        assert(test.every(v => !this.train.some(t => group(t) === group(v))))
        seen.push(test)
        return new Float64Array(Xtest.rows)
      }
      dispose() {}
    }
    const search = new Search([{ name: 'model', cls: Model, searchSpace: {} }], {
      nIter: 1, cv: plan, task: 'regression', scoring: 'mse'
    })
    const { bestResult } = await search.fit(X, yReg)
    assert.equal(bestResult.candidate.model.params.task, 'regression')
    assert(seen.length > 0)
    const fitted = await search.refitBest(X, yReg)
    fitted.dispose()
  })
}

describe('RandomSearch classification', () => {
  it('fits and returns leaderboard', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 5, cv: 3, task: 'classification' }
    )
    const { leaderboard, bestResult } = await search.fit(X, yCls)
    assert(leaderboard.length > 0)
    assert(bestResult !== null)
    assert(bestResult.meanScore >= 0 && bestResult.meanScore <= 1)
  })

  it('has correct number of entries (nModels * nIter)', async () => {
    const search = new RandomSearch(
      [
        { name: 'm1', classId: 'wlearn.test.m1@1', cls: SearchableMock },
        { name: 'm2', classId: 'wlearn.test.m2@1', cls: SearchableMock },
      ],
      { nIter: 3, cv: 2, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    assert.equal(leaderboard.length, 6) // 2 models * 3 iter
  })

  it('best result has rank 1', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 5, cv: 2, task: 'classification' }
    )
    const { bestResult } = await search.fit(X, yCls)
    assert.equal(bestResult.rank, 1)
  })

  it('all entries have per-fold scores', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 3, cv: 4, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    for (const entry of leaderboard.ranked()) {
      assert.equal(entry.scores.length, 4)
    }
  })

  it('refitBest returns a fitted model', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 3, cv: 2, task: 'classification' }
    )
    await search.fit(X, yCls)
    const model = await search.refitBest(X, yCls)
    assert(model.isFitted)
    const preds = model.predict(X)
    assert.equal(preds.length, 10)
    model.dispose()
  })

  it('deterministic with same seed', async () => {
    const run = async (seed) => {
      const search = new RandomSearch(
        [{ name: 'mock', cls: SearchableMock }],
        { nIter: 5, cv: 3, seed, task: 'classification' }
      )
      const { leaderboard } = await search.fit(X, yCls)
      return leaderboard.ranked().map(e => e.meanScore)
    }
    const a = await run(42)
    const b = await run(42)
    assert.deepEqual(a, b)
  })
})

describe('RandomSearch regression', () => {
  it('fits and returns leaderboard', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMockReg }],
      { nIter: 3, cv: 3, task: 'regression' }
    )
    const { leaderboard, bestResult } = await search.fit(X, yReg)
    assert(leaderboard.length > 0)
    assert(isFinite(bestResult.meanScore))
  })
})

describe('RandomSearch options', () => {
  it('accepts custom scoring function', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 2, cv: 2, task: 'classification', scoring: () => 0.42 }
    )
    const { bestResult } = await search.fit(X, yCls)
    assert(Math.abs(bestResult.meanScore - 0.42) < 1e-10)
  })

  it('accepts custom search space override', async () => {
    const search = new RandomSearch(
      [{
        name: 'mock', cls: SearchableMock,
        searchSpace: { bias: { type: 'categorical', values: [0] } },
        params: { task: 'classification' },
      }],
      { nIter: 3, cv: 2, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    // All entries should have bias=0
    for (const e of leaderboard.ranked()) {
      assert.equal(e.params.bias, 0)
    }
  })

  it('fixed params override search space', async () => {
    const search = new RandomSearch(
      [{
        name: 'mock', cls: SearchableMock,
        params: { bias: 99, task: 'classification' },
      }],
      { nIter: 3, cv: 2, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    for (const e of leaderboard.ranked()) {
      assert.equal(e.params.bias, 99)
      assert.equal(e.params.task, 'classification')
    }
  })

  it('throws on empty models', () => {
    assert.throws(() => new RandomSearch([]), ValidationError)
  })

  it('throws if refitBest called before fit', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 1, cv: 2 }
    )
    await assert.rejects(() => search.refitBest(X, yCls), ValidationError)
  })
})
