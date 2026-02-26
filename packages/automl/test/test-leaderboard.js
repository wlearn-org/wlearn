import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { Leaderboard } from '../src/leaderboard.js'

function makeScores(...vals) {
  return new Float64Array(vals)
}

describe('Leaderboard', () => {
  it('add creates entry with id', () => {
    const lb = new Leaderboard()
    const e = lb.add({ modelName: 'm1', params: { a: 1 }, scores: makeScores(0.8, 0.9), fitTimeMs: 100 })
    assert.equal(e.id, 0)
    assert.equal(e.modelName, 'm1')
    assert.equal(lb.length, 1)
  })

  it('computes meanScore and stdScore correctly', () => {
    const lb = new Leaderboard()
    const e = lb.add({ modelName: 'm1', params: {}, scores: makeScores(0.6, 0.8, 1.0), fitTimeMs: 50 })
    assert(Math.abs(e.meanScore - 0.8) < 1e-10)
    // std = sqrt(((0.2^2 + 0 + 0.2^2) / 3)) = sqrt(0.08/3) ~= 0.1633
    assert(Math.abs(e.stdScore - Math.sqrt(0.08 / 3)) < 1e-10)
  })

  it('ranked sorts by meanScore descending', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'low', params: {}, scores: makeScores(0.5), fitTimeMs: 10 })
    lb.add({ modelName: 'high', params: {}, scores: makeScores(0.9), fitTimeMs: 10 })
    lb.add({ modelName: 'mid', params: {}, scores: makeScores(0.7), fitTimeMs: 10 })
    const r = lb.ranked()
    assert.equal(r[0].modelName, 'high')
    assert.equal(r[1].modelName, 'mid')
    assert.equal(r[2].modelName, 'low')
  })

  it('ranked assigns correct ranks', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'a', params: {}, scores: makeScores(0.3), fitTimeMs: 10 })
    lb.add({ modelName: 'b', params: {}, scores: makeScores(0.9), fitTimeMs: 10 })
    const r = lb.ranked()
    assert.equal(r[0].rank, 1)
    assert.equal(r[1].rank, 2)
  })

  it('best returns highest-scoring entry', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'a', params: {}, scores: makeScores(0.3), fitTimeMs: 10 })
    lb.add({ modelName: 'b', params: {}, scores: makeScores(0.9), fitTimeMs: 10 })
    assert.equal(lb.best().modelName, 'b')
  })

  it('best returns null when empty', () => {
    const lb = new Leaderboard()
    assert.equal(lb.best(), null)
  })

  it('top(k) returns k entries', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'a', params: {}, scores: makeScores(0.3), fitTimeMs: 10 })
    lb.add({ modelName: 'b', params: {}, scores: makeScores(0.9), fitTimeMs: 10 })
    lb.add({ modelName: 'c', params: {}, scores: makeScores(0.6), fitTimeMs: 10 })
    const t = lb.top(2)
    assert.equal(t.length, 2)
    assert.equal(t[0].modelName, 'b')
    assert.equal(t[1].modelName, 'c')
  })

  it('toJSON and fromJSON round-trip', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'a', params: { x: 1 }, scores: makeScores(0.8, 0.9), fitTimeMs: 42 })
    lb.add({ modelName: 'b', params: { y: 2 }, scores: makeScores(0.7, 0.6), fitTimeMs: 33 })

    const json = lb.toJSON()
    const lb2 = Leaderboard.fromJSON(json)
    assert.equal(lb2.length, 2)
    assert.equal(lb2.best().modelName, 'a')
    assert(lb2.best().scores instanceof Float64Array)
  })

  it('length returns correct count', () => {
    const lb = new Leaderboard()
    assert.equal(lb.length, 0)
    lb.add({ modelName: 'a', params: {}, scores: makeScores(0.5), fitTimeMs: 10 })
    assert.equal(lb.length, 1)
    lb.add({ modelName: 'b', params: {}, scores: makeScores(0.6), fitTimeMs: 10 })
    assert.equal(lb.length, 2)
  })

  it('adding after ranked re-sorts correctly', () => {
    const lb = new Leaderboard()
    lb.add({ modelName: 'a', params: {}, scores: makeScores(0.5), fitTimeMs: 10 })
    lb.ranked() // trigger sort
    lb.add({ modelName: 'b', params: {}, scores: makeScores(0.9), fitTimeMs: 10 })
    assert.equal(lb.best().modelName, 'b')
  })
})
