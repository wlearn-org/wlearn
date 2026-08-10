const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const {
  detectTask, partialShuffle, scorerGreaterIsBetter
} = require('../src/common.js')
const {
  candidateCanonicalBytes, candidateHash, createCandidate,
  makeCandidateId, normalizeModelSpecs, seedFor
} = require('../src/candidate.js')

const vectors = JSON.parse(fs.readFileSync(
  path.join(__dirname, 'candidate-v1.json'), 'utf8'
))

describe('detectTask', () => {
  it('classifies Int32Array as classification', () => {
    assert.equal(detectTask(new Int32Array([0, 1, 0, 1])), 'classification')
  })

  it('classifies non-integer Float64Array as regression', () => {
    assert.equal(detectTask(new Float64Array([1.1, 2.2, 3.3])), 'regression')
  })

  it('classifies few unique integers as classification', () => {
    assert.equal(detectTask(new Float64Array([0, 1, 2, 0, 1, 2])), 'classification')
  })

  it('classifies many unique integers as regression', () => {
    const y = new Float64Array(100)
    for (let i = 0; i < 100; i++) y[i] = i
    assert.equal(detectTask(y), 'regression')
  })
})

describe('candidate identity v1', () => {
  for (const vector of vectors.cases) {
    it(`matches shared canonical/hash/seed vector ${vector.name}`, () => {
      const candidate = createCandidate(
        vector.model, vector.params, vector.preprocess
      )
      assert.equal(
        Buffer.from(candidateCanonicalBytes(candidate)).toString('utf8'),
        vector.canonicalUtf8
      )
      assert.equal(candidateHash(candidate), vector.sha256)
      assert.equal(makeCandidateId(candidate), vector.candidateId)
      for (const seed of vector.seeds) {
        assert.equal(
          seedFor(candidate, seed.foldId, seed.baseSeed), seed.value
        )
      }
    })
  }

  it('ignores display-name changes but distinguishes class and preprocessing', () => {
    const first = createCandidate(
      { displayName: 'first', classId: 'wlearn.test.a@1' }, { x: 1 }
    )
    const renamed = createCandidate(
      { displayName: 'renamed', classId: 'wlearn.test.a@1' }, { x: 1 }
    )
    const otherClass = createCandidate(
      { displayName: 'first', classId: 'wlearn.test.b@1' }, { x: 1 }
    )
    const withPreprocess = createCandidate(
      first.model, first.model.params, vectors.cases[1].preprocess
    )
    assert.equal(makeCandidateId(first), makeCandidateId(renamed))
    assert.notEqual(makeCandidateId(first), makeCandidateId(otherClass))
    assert.notEqual(makeCandidateId(first), makeCandidateId(withPreprocess))
  })

  it('rejects nonportable values, missing IDs, and duplicate class IDs', () => {
    assert.throws(() => createCandidate(
      { displayName: 'x', classId: 'wlearn.test.x@1' }, { x: NaN }
    ), /finite/)
    assert.throws(() => createCandidate(
      { displayName: 'x', classId: 'wlearn.test.x@1' }, { x: 2 ** 53 }
    ), /safe-integer/)
    assert.throws(() => normalizeModelSpecs([
      { name: 'x', cls: { create() {} } }
    ]), /classId/)
    const cls = { classId: 'wlearn.test.same@1', create() {} }
    assert.throws(() => normalizeModelSpecs([
      { name: 'x', cls }, { name: 'y', cls }
    ]), /duplicate/)
  })

  it('rejects ambiguous containers and non-object model params', () => {
    const model = { displayName: 'x', classId: 'wlearn.test.x@1' }
    const sparse = new Array(1)
    const cyclic = {}
    cyclic.self = cyclic
    for (const params of [sparse, [], null, 1, 'x']) {
      assert.throws(() => createCandidate(model, params), /plain object/)
    }
    assert.throws(() => createCandidate(model, { sparse }), /sparse/)
    assert.throws(() => createCandidate(model, { missing: undefined }), /portable JSON/)
    assert.throws(() => createCandidate(model, cyclic), /cyclic/)
    assert.throws(() => normalizeModelSpecs([
      { name: 'x', cls: { classId: model.classId, create() {} }, params: [] }
    ]), /params.*plain object/)
    assert.throws(() => normalizeModelSpecs([{
      name: 'x', cls: { classId: model.classId, create() {} },
      preprocessChoices: null,
    }]), /preprocessChoices.*nonempty array/)
  })

  it('preserves adversarial keys and resolves preprocessing before identity', () => {
    const params = JSON.parse('{"__proto__":{"safe":true}}')
    const model = { displayName: 'x', classId: 'wlearn.test.x@1' }
    const partial = createCandidate(model, params, {
      templateId: 'default',
      typeId: 'wlearn.preprocess.tabular@1',
      resolvedParams: {},
    })
    const explicit = createCandidate(model, params, {
      templateId: 'default',
      typeId: 'wlearn.preprocess.tabular@1',
      resolvedParams: partial.preprocess.resolvedParams,
    })
    assert(Object.hasOwn(partial.model.params, '__proto__'))
    assert.equal(makeCandidateId(partial), makeCandidateId(explicit))
  })
})

describe('partialShuffle', () => {
  it('selects k elements from array', () => {
    const rng = () => 0.5
    const arr = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    const result = partialShuffle(arr, 3, rng)
    assert.equal(result.length, 3)
  })

  it('returns all elements when k >= n', () => {
    const rng = () => 0.5
    const arr = new Int32Array([0, 1, 2])
    const result = partialShuffle(arr, 5, rng)
    assert.equal(result.length, 3)
  })

  it('is deterministic with same rng', () => {
    const makeRng = () => {
      let s = 42
      return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff }
    }
    const a = partialShuffle(new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 4, makeRng())
    const b = partialShuffle(new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 4, makeRng())
    assert.deepEqual([...a], [...b])
  })

  it('selected elements are from original array', () => {
    const rng = () => 0.3
    const arr = new Int32Array([10, 20, 30, 40, 50])
    const result = partialShuffle(arr, 3, rng)
    const original = new Set([10, 20, 30, 40, 50])
    for (const v of result) assert(original.has(v))
  })

  it('selected elements are unique', () => {
    const rng = () => 0.7
    const arr = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    const result = partialShuffle(arr, 5, rng)
    const unique = new Set(result)
    assert.equal(unique.size, result.length)
  })
})

describe('scorerGreaterIsBetter', () => {
  it('returns true for accuracy', () => {
    assert.equal(scorerGreaterIsBetter('accuracy'), true)
  })

  it('returns true for r2', () => {
    assert.equal(scorerGreaterIsBetter('r2'), true)
  })

  it('returns true for neg_mse', () => {
    assert.equal(scorerGreaterIsBetter('neg_mse'), true)
  })

  it('returns true for custom function', () => {
    assert.equal(scorerGreaterIsBetter(() => 0.5), true)
  })
})
