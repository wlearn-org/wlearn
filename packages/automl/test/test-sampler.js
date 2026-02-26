import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { sampleParam, sampleConfig, randomConfigs, gridConfigs } from '../src/sampler.js'
import { makeLCG } from '@wlearn/core'

describe('sampleParam', () => {
  it('categorical returns value from list', () => {
    const rng = makeLCG(1)
    const vals = new Set([10, 20, 30])
    for (let i = 0; i < 50; i++) {
      const v = sampleParam({ type: 'categorical', values: [10, 20, 30] }, rng)
      assert(vals.has(v), `unexpected: ${v}`)
    }
  })

  it('uniform returns value in range', () => {
    const rng = makeLCG(2)
    for (let i = 0; i < 50; i++) {
      const v = sampleParam({ type: 'uniform', low: 1.0, high: 5.0 }, rng)
      assert(v >= 1.0 && v <= 5.0, `out of range: ${v}`)
    }
  })

  it('log_uniform returns value in range with log spacing', () => {
    const rng = makeLCG(3)
    const vals = []
    for (let i = 0; i < 1000; i++) {
      const v = sampleParam({ type: 'log_uniform', low: 0.001, high: 1000 }, rng)
      assert(v >= 0.001 && v <= 1000, `out of range: ${v}`)
      vals.push(v)
    }
    // Median should be near geometric mean = sqrt(0.001 * 1000) = 1.0
    vals.sort((a, b) => a - b)
    const median = vals[500]
    assert(median > 0.1 && median < 10, `median ${median} not near geometric mean`)
  })

  it('int_uniform returns integers', () => {
    const rng = makeLCG(4)
    for (let i = 0; i < 50; i++) {
      const v = sampleParam({ type: 'int_uniform', low: 3, high: 10 }, rng)
      assert(v >= 3 && v <= 10, `out of range: ${v}`)
      assert.equal(v, Math.floor(v), `not integer: ${v}`)
    }
  })

  it('int_log_uniform returns integers', () => {
    const rng = makeLCG(5)
    for (let i = 0; i < 50; i++) {
      const v = sampleParam({ type: 'int_log_uniform', low: 1, high: 100 }, rng)
      assert(v >= 1 && v <= 100, `out of range: ${v}`)
      assert.equal(v, Math.round(v), `not integer: ${v}`)
    }
  })

  it('throws on unknown type', () => {
    const rng = makeLCG(6)
    assert.throws(() => sampleParam({ type: 'banana' }, rng), /Unknown/)
  })
})

describe('sampleConfig', () => {
  it('samples all non-conditional params', () => {
    const rng = makeLCG(7)
    const space = {
      a: { type: 'uniform', low: 0, high: 1 },
      b: { type: 'categorical', values: ['x', 'y'] },
    }
    const cfg = sampleConfig(space, rng)
    assert('a' in cfg)
    assert('b' in cfg)
  })

  it('includes conditional param when condition met', () => {
    const rng = makeLCG(100)
    // Force kernel to POLY by making it the only option
    const space = {
      kernel: { type: 'categorical', values: ['POLY'] },
      degree: { type: 'int_uniform', low: 2, high: 5, condition: { kernel: 'POLY' } },
    }
    const cfg = sampleConfig(space, rng)
    assert.equal(cfg.kernel, 'POLY')
    assert('degree' in cfg)
    assert(cfg.degree >= 2 && cfg.degree <= 5)
  })

  it('omits conditional param when condition not met', () => {
    const rng = makeLCG(101)
    const space = {
      kernel: { type: 'categorical', values: ['RBF'] },
      degree: { type: 'int_uniform', low: 2, high: 5, condition: { kernel: 'POLY' } },
    }
    const cfg = sampleConfig(space, rng)
    assert.equal(cfg.kernel, 'RBF')
    assert(!('degree' in cfg))
  })

  it('empty space returns empty config', () => {
    const rng = makeLCG(8)
    const cfg = sampleConfig({}, rng)
    assert.deepEqual(cfg, {})
  })
})

describe('randomConfigs', () => {
  const space = {
    a: { type: 'uniform', low: 0, high: 1 },
    b: { type: 'int_uniform', low: 1, high: 10 },
  }

  it('returns n configs', () => {
    const cfgs = randomConfigs(space, 10)
    assert.equal(cfgs.length, 10)
  })

  it('deterministic with same seed', () => {
    const a = randomConfigs(space, 5, { seed: 42 })
    const b = randomConfigs(space, 5, { seed: 42 })
    assert.deepEqual(a, b)
  })

  it('different with different seed', () => {
    const a = randomConfigs(space, 5, { seed: 1 })
    const b = randomConfigs(space, 5, { seed: 2 })
    // At least one config should differ
    const same = a.every((cfg, i) => cfg.a === b[i].a && cfg.b === b[i].b)
    assert(!same, 'all configs identical with different seeds')
  })
})

describe('gridConfigs', () => {
  it('enumerates all combinations for categorical', () => {
    const space = {
      a: { type: 'categorical', values: ['x', 'y'] },
      b: { type: 'categorical', values: [1, 2, 3] },
    }
    const cfgs = gridConfigs(space)
    assert.equal(cfgs.length, 6) // 2 * 3
  })

  it('discretizes continuous params', () => {
    const space = {
      a: { type: 'uniform', low: 0, high: 1 },
    }
    const cfgs = gridConfigs(space, { steps: 3 })
    assert.equal(cfgs.length, 3)
    assert.equal(cfgs[0].a, 0)
    assert.equal(cfgs[1].a, 0.5)
    assert.equal(cfgs[2].a, 1)
  })

  it('handles int_uniform with small range', () => {
    const space = {
      k: { type: 'int_uniform', low: 1, high: 3 },
    }
    const cfgs = gridConfigs(space)
    assert.equal(cfgs.length, 3) // 1, 2, 3
    assert.deepEqual(cfgs.map(c => c.k), [1, 2, 3])
  })

  it('empty space returns single empty config', () => {
    const cfgs = gridConfigs({})
    assert.equal(cfgs.length, 1)
    assert.deepEqual(cfgs[0], {})
  })
})
