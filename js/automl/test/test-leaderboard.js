const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { Leaderboard } = require('../src/leaderboard.js')
const { createCandidate, makeCandidateId } = require('../src/candidate.js')

function makeScores(...vals) {
  return new Float64Array(vals)
}

function makeEntry(name, params, scores, fitTimeMs) {
  const candidate = createCandidate({
    displayName: name,
    classId: `wlearn.test.${name}@1`
  }, params)
  return {
    candidateId: makeCandidateId(candidate), candidate, scores, fitTimeMs
  }
}

describe('Leaderboard', () => {
  it('add creates entry with id', () => {
    const lb = new Leaderboard()
    const e = lb.add(makeEntry('m1', { a: 1 }, makeScores(0.8, 0.9), 100))
    assert.equal(e.id, 0)
    assert.equal(e.modelName, 'm1')
    assert.equal(lb.length, 1)
  })

  it('computes meanScore and stdScore correctly', () => {
    const lb = new Leaderboard()
    const e = lb.add(makeEntry('m1', {}, makeScores(0.6, 0.8, 1.0), 50))
    assert(Math.abs(e.meanScore - 0.8) < 1e-10)
    // std = sqrt(((0.2^2 + 0 + 0.2^2) / 3)) = sqrt(0.08/3) ~= 0.1633
    assert(Math.abs(e.stdScore - Math.sqrt(0.08 / 3)) < 1e-10)
  })

  it('ranked sorts by meanScore descending', () => {
    const lb = new Leaderboard()
    lb.add(makeEntry('low', {}, makeScores(0.5), 10))
    lb.add(makeEntry('high', {}, makeScores(0.9), 10))
    lb.add(makeEntry('mid', {}, makeScores(0.7), 10))
    const r = lb.ranked()
    assert.equal(r[0].modelName, 'high')
    assert.equal(r[1].modelName, 'mid')
    assert.equal(r[2].modelName, 'low')
  })

  it('ranked assigns correct ranks', () => {
    const lb = new Leaderboard()
    lb.add(makeEntry('a', {}, makeScores(0.3), 10))
    lb.add(makeEntry('b', {}, makeScores(0.9), 10))
    const r = lb.ranked()
    assert.equal(r[0].rank, 1)
    assert.equal(r[1].rank, 2)
  })

  it('best returns highest-scoring entry', () => {
    const lb = new Leaderboard()
    lb.add(makeEntry('a', {}, makeScores(0.3), 10))
    lb.add(makeEntry('b', {}, makeScores(0.9), 10))
    assert.equal(lb.best().modelName, 'b')
  })

  it('best returns null when empty', () => {
    const lb = new Leaderboard()
    assert.equal(lb.best(), null)
  })

  it('top(k) returns k entries', () => {
    const lb = new Leaderboard()
    lb.add(makeEntry('a', {}, makeScores(0.3), 10))
    lb.add(makeEntry('b', {}, makeScores(0.9), 10))
    lb.add(makeEntry('c', {}, makeScores(0.6), 10))
    const t = lb.top(2)
    assert.equal(t.length, 2)
    assert.equal(t[0].modelName, 'b')
    assert.equal(t[1].modelName, 'c')
  })

  it('toJSON and fromJSON round-trip', () => {
    const lb = new Leaderboard()
    const first = makeEntry('a', { x: 1 }, makeScores(0.8, 0.9), 42)
    lb.add(first)
    lb.add(makeEntry('b', { y: 2 }, makeScores(0.7, 0.6), 33))

    const json = lb.toJSON()
    const lb2 = Leaderboard.fromJSON(json)
    assert.equal(lb2.length, 2)
    assert.equal(lb2.best().modelName, 'a')
    assert.equal(lb2.best().candidateId, first.candidateId)
    assert(lb2.best().scores instanceof Float64Array)
  })

  it('snapshots params and scores on insert and JSON export', () => {
    const lb = new Leaderboard()
    const params = { depth: 3, nested: { eta: 0.1 } }
    const scores = makeScores(0.8, 0.9)
    lb.add(makeEntry('xgb', params, scores, 10))

    params.nested.eta = 9
    scores[0] = 0
    assert.equal(lb.best().params.nested.eta, 0.1)
    assert.equal(lb.best().scores[0], 0.8)

    const best = lb.best()
    best.params.nested.eta = 5
    best.scores[0] = 0
    assert.equal(lb.best().params.nested.eta, 0.1)
    assert.equal(lb.best().scores[0], 0.8)

    const json = lb.toJSON()
    json[0].params.nested.eta = 7
    assert.equal(lb.best().params.nested.eta, 0.1)
  })

  it('converts entries to Archive', () => {
    const lb = new Leaderboard()
    const source = makeEntry('a', {}, makeScores(0.8, 0.9), 42)
    lb.add(source)

    const archive = lb.toArchive({ metric: 'accuracy' })
    assert.equal(archive.size, 1)
    const record = archive.records()[0]
    assert.equal(record.status, 'ok')
    assert(Math.abs(record.scores.accuracy - 0.85) < 1e-12)
    assert.equal(record.metadata.sourceCandidateId, source.candidateId)
    assert.deepEqual(record.metadata.candidate, source.candidate)
  })

  it('length returns correct count', () => {
    const lb = new Leaderboard()
    assert.equal(lb.length, 0)
    lb.add(makeEntry('a', {}, makeScores(0.5), 10))
    assert.equal(lb.length, 1)
    lb.add(makeEntry('b', {}, makeScores(0.6), 10))
    assert.equal(lb.length, 2)
  })

  it('adding after ranked re-sorts correctly', () => {
    const lb = new Leaderboard()
    lb.add(makeEntry('a', {}, makeScores(0.5), 10))
    lb.ranked() // trigger sort
    lb.add(makeEntry('b', {}, makeScores(0.9), 10))
    assert.equal(lb.best().modelName, 'b')
  })
})
