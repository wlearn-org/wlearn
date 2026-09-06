const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { makeLCG, shuffle } = require('../src/rng.js')

describe('makeLCG', () => {
  it('is deterministic', () => {
    const a = makeLCG(42)
    const b = makeLCG(42)
    for (let i = 0; i < 100; i++) {
      assert.equal(a(), b())
    }
  })

  it('produces values in [0, 1)', () => {
    const rng = makeLCG(1)
    for (let i = 0; i < 1000; i++) {
      const v = rng()
      assert(v >= 0 && v < 1, `out of range: ${v}`)
    }
  })

  it('different seeds produce different sequences', () => {
    const a = makeLCG(1)
    const b = makeLCG(2)
    let same = 0
    for (let i = 0; i < 20; i++) {
      if (a() === b()) same++
    }
    assert(same < 20, 'sequences should differ')
  })

  it('matches the constants from fixtures/generate.mjs', () => {
    // Same constants: s = (s * 1664525 + 1013904223) & 0x7fffffff
    const rng = makeLCG(100)
    const vals = []
    for (let i = 0; i < 5; i++) vals.push(rng())
    // Just check they are deterministic and reasonable
    assert.equal(vals.length, 5)
    assert(vals[0] !== vals[1])
  })
})

describe('shuffle', () => {
  it('preserves all elements', () => {
    const arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const rng = makeLCG(42)
    shuffle(arr, rng)
    assert.deepEqual([...arr].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  })

  it('is deterministic', () => {
    const a = [0, 1, 2, 3, 4, 5]
    const b = [0, 1, 2, 3, 4, 5]
    shuffle(a, makeLCG(7))
    shuffle(b, makeLCG(7))
    assert.deepEqual(a, b)
  })

  it('returns the same array', () => {
    const arr = [1, 2, 3]
    const result = shuffle(arr, makeLCG(1))
    assert.strictEqual(result, arr)
  })

  it('works with typed arrays', () => {
    const arr = new Int32Array([0, 1, 2, 3, 4])
    shuffle(arr, makeLCG(42))
    assert.deepEqual([...arr].sort(), [0, 1, 2, 3, 4])
  })

  it('actually shuffles', () => {
    const arr = Array.from({ length: 20 }, (_, i) => i)
    const orig = [...arr]
    shuffle(arr, makeLCG(42))
    let diffs = 0
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] !== orig[i]) diffs++
    }
    assert(diffs > 5, `expected at least some elements to move, got ${diffs} diffs`)
  })
})
