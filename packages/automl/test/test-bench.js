const { test } = require('node:test')
const assert = require('node:assert/strict')

async function run(model, specs = [{}]) {
  const { runWlearn } = await import('../bench/bench-automl.mjs')
  return runWlearn(specs, [[0], [1]], [0, 1], [[0], [1]], [0, 1], 'classification', 'random', {
    autoFitFn: async () => ({ model }),
  })
}

test('benchmark awaits inference and records the fitted artifact', async () => {
  let disposed = 0
  const result = await run({
    predict: async () => new Int32Array([0, 1]),
    save: () => new Uint8Array([1, 2, 3]),
    dispose: () => disposed++,
  })
  assert.equal(result.score, 1)
  assert.equal(result.bundle_bytes, 3)
  assert.equal(disposed, 1)
})

test('benchmark propagates inference failure and always disposes the model', async () => {
  let disposed = 0
  await assert.rejects(run({
    predict: () => { throw new Error('inference failed') },
    dispose: () => disposed++,
  }), /inference failed/)
  assert.equal(disposed, 1)
})

test('benchmark rejects missing candidates and missing fitted models', async () => {
  await assert.rejects(run(null, []), /model|candidate/i)
  await assert.rejects(run(null), /model/i)
})
