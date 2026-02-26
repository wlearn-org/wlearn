import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { PORTFOLIO, getPortfolio, PortfolioStrategy, PortfolioSearch } from '../src/portfolio.js'
import { autoFit } from '../src/auto-fit.js'
import { SearchableMock, SearchableMockReg } from './mock-model.js'
import { ValidationError } from '@wlearn/core'

describe('getPortfolio', () => {
  it('returns classification portfolio', () => {
    const p = getPortfolio('classification')
    assert(p.xgb && p.xgb.length > 0)
    assert(p.ebm && p.ebm.length > 0)
    assert(p.linear && p.linear.length > 0)
    assert(p.svm && p.svm.length > 0)
    assert(p.knn && p.knn.length > 0)
    assert(p.tsetlin && p.tsetlin.length > 0)
  })

  it('returns regression portfolio', () => {
    const p = getPortfolio('regression')
    assert(p.xgb && p.xgb.length > 0)
    assert(p.ebm && p.ebm.length > 0)
    assert(p.linear && p.linear.length > 0)
    assert(p.svm && p.svm.length > 0)
    assert(p.knn && p.knn.length > 0)
    assert(p.tsetlin && p.tsetlin.length > 0)
  })

  it('has correct config counts', () => {
    const cls = getPortfolio('classification')
    assert.equal(cls.xgb.length, 10)
    assert.equal(cls.ebm.length, 4)
    assert.equal(cls.linear.length, 4)
    assert.equal(cls.svm.length, 4)
    assert.equal(cls.knn.length, 3)
    assert.equal(cls.tsetlin.length, 3)
  })

  it('xgb classification configs have multi:softprob objective', () => {
    const cls = getPortfolio('classification')
    for (const c of cls.xgb) {
      assert.equal(c.objective, 'multi:softprob')
    }
  })

  it('xgb regression configs have reg:squarederror objective', () => {
    const reg = getPortfolio('regression')
    for (const c of reg.xgb) {
      assert.equal(c.objective, 'reg:squarederror')
    }
  })

  it('ebm classification configs have classification objective', () => {
    for (const c of getPortfolio('classification').ebm) {
      assert.equal(c.objective, 'classification')
    }
  })

  it('ebm regression configs have regression objective', () => {
    for (const c of getPortfolio('regression').ebm) {
      assert.equal(c.objective, 'regression')
    }
  })

  it('falls back to classification for unknown task', () => {
    const p = getPortfolio('unknown')
    assert.deepEqual(p, PORTFOLIO.classification)
  })

  it('classification and regression have same model families', () => {
    const cls = Object.keys(getPortfolio('classification')).sort()
    const reg = Object.keys(getPortfolio('regression')).sort()
    assert.deepEqual(cls, reg)
  })
})

describe('PortfolioStrategy', () => {
  it('yields all portfolio configs for known models', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'xgb', cls: SearchableMock }],
      { task: 'classification' }
    )
    let count = 0
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      count++
    }
    assert.equal(count, 10) // 10 XGB configs (8 boosting + 2 RF-mode)
  })

  it('yields correct total for multiple models', () => {
    const strategy = new PortfolioStrategy(
      [
        { name: 'xgb', cls: SearchableMock },
        { name: 'knn', cls: SearchableMock },
      ],
      { task: 'classification' }
    )
    let count = 0
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      count++
    }
    assert.equal(count, 13) // 10 XGB + 3 KNN
  })

  it('fixed params override portfolio configs', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'xgb', cls: SearchableMock, params: { eta: 999 } }],
      { task: 'classification' }
    )
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      assert.equal(task.params.eta, 999)
    }
  })

  it('falls back to single default config for unknown model', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'unknown_model', cls: SearchableMock }],
      { task: 'classification' }
    )
    let count = 0
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      count++
    }
    assert.equal(count, 1)
  })

  it('returns null after exhaustion', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'knn', cls: SearchableMock }],
      { task: 'classification' }
    )
    // Drain all
    while (!strategy.isDone()) strategy.next()
    assert.equal(strategy.next(), null)
    assert.equal(strategy.isDone(), true)
  })

  it('candidates have candidateId, cls, params', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'knn', cls: SearchableMock }],
      { task: 'classification' }
    )
    const task = strategy.next()
    assert(typeof task.candidateId === 'string')
    assert.equal(task.cls, SearchableMock)
    assert(typeof task.params === 'object')
    assert(task.params.k !== undefined)
  })

  it('report is a no-op', () => {
    const strategy = new PortfolioStrategy(
      [{ name: 'knn', cls: SearchableMock }],
      { task: 'classification' }
    )
    strategy.report({ candidateId: 'x', meanScore: 0.5 })
  })
})

describe('PortfolioSearch', () => {
  const X = {
    data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    rows: 10, cols: 2
  }
  const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
  const yReg = new Float64Array([1.1, 2.3, 3.7, 4.2, 5.8, 6.1, 7.5, 8.9, 9.4, 10.6])

  it('classification: fit returns leaderboard', async () => {
    const search = new PortfolioSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { cv: 2, task: 'classification' }
    )
    const { leaderboard, bestResult } = await search.fit(X, yCls)
    // mock model has 1 portfolio entry (falls back to default)
    assert(leaderboard.length >= 1)
    assert(bestResult.meanScore >= 0)
  })

  it('regression: fit returns leaderboard', async () => {
    const search = new PortfolioSearch(
      [{ name: 'mock', cls: SearchableMockReg }],
      { cv: 2, task: 'regression' }
    )
    const { leaderboard, bestResult } = await search.fit(X, yReg)
    assert(leaderboard.length >= 1)
    assert(isFinite(bestResult.meanScore))
  })

  it('refitBest returns fitted model', async () => {
    const search = new PortfolioSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { cv: 2, task: 'classification' }
    )
    await search.fit(X, yCls)
    const model = await search.refitBest(X, yCls)
    assert(model.isFitted)
    const preds = model.predict(X)
    assert.equal(preds.length, 10)
    model.dispose()
  })

  it('throws on empty models', () => {
    assert.throws(
      () => new PortfolioSearch([], { task: 'classification' }),
      ValidationError
    )
  })

  it('throws if refitBest called before fit', async () => {
    const search = new PortfolioSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { cv: 2, task: 'classification' }
    )
    await assert.rejects(
      () => search.refitBest(X, yCls),
      ValidationError
    )
  })
})

describe('autoFit with strategy=portfolio', () => {
  const X = {
    data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    rows: 10, cols: 2
  }
  const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

  it('strategy=portfolio uses PortfolioSearch', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { cv: 2, task: 'classification', strategy: 'portfolio' }
    )
    assert(result.model !== null)
    assert(result.model.isFitted)
    assert(result.leaderboard.length >= 1)
    result.model.dispose()
  })

  it('strategy=portfolio with ensemble', async () => {
    const result = await autoFit(
      [
        { name: 'm1', cls: SearchableMock },
        { name: 'm2', cls: SearchableMock },
      ],
      X, yCls,
      { cv: 2, task: 'classification', strategy: 'portfolio', ensemble: true, ensembleSize: 3 }
    )
    assert(result.model !== null)
    assert(result.model.capabilities.classifier)
    result.model.dispose()
  })
})
