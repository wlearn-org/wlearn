/**
 * Tests for @wlearn/nn (MLPClassifier, MLPRegressor)
 */

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol = 1e-5, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// ── Test Data Generators ───────────────────────────────────────────

// Simple seeded PRNG (LCG)
function lcg(seed) {
  let s = seed
  return function () {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeBinaryData(seed = 42, n = 50, nFeatures = 2) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push((rng() - 0.5) * 4) // roughly randn-like
    }
    X.push(row)
    y.push(row[0] + row[1] > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeRegressionData(seed = 42, n = 50, nFeatures = 2) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push((rng() - 0.5) * 4)
    }
    X.push(row)
    y.push(2.0 * row[0] + 3.0 * row[1] + 0.5)
  }
  return { X, y }
}

// ── Import ─────────────────────────────────────────────────────────

const { MLPClassifier, MLPRegressor } = await import('../src/index.js')
const { load, decodeBundle } = await import('@wlearn/core')

// ═══════════════════════════════════════════════════════════════════
// MLPClassifier
// ═══════════════════════════════════════════════════════════════════

console.log('\n=== MLPClassifier ===')

await test('create() returns unfitted model', async () => {
  const model = await MLPClassifier.create()
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('throws before fit', async () => {
  const model = await MLPClassifier.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')
  model.dispose()
})

await test('binary classification', async () => {
  const { X, y } = makeBinaryData()
  const model = await MLPClassifier.create({
    hidden_sizes: [8], epochs: 50, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.score(X, y) > 0.6, `accuracy should be > 0.6, got ${model.score(X, y)}`)
  model.dispose()
})

await test('predict returns Float64Array with correct length', async () => {
  const { X, y } = makeBinaryData(42, 20)
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 10, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'should be Float64Array')
  assert(preds.length === 20, `expected 20, got ${preds.length}`)
  model.dispose()
})

await test('predictProba returns valid probabilities', async () => {
  const { X, y } = makeBinaryData(42, 20)
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 10, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  const proba = model.predictProba(X)
  assert(proba instanceof Float64Array, 'should be Float64Array')
  assert(proba.length === 20 * 2, `expected 40, got ${proba.length}`)

  // Check probabilities sum to 1 per sample and are non-negative
  for (let i = 0; i < 20; i++) {
    const sum = proba[i * 2] + proba[i * 2 + 1]
    assertClose(sum, 1.0, 1e-5, `proba sum at ${i} = ${sum}`)
    assert(proba[i * 2] >= 0, 'proba should be >= 0')
    assert(proba[i * 2 + 1] >= 0, 'proba should be >= 0')
  }
  model.dispose()
})

await test('classes property', async () => {
  const { X, y } = makeBinaryData()
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  const classes = model.classes
  assert(classes.length === 2, `expected 2 classes, got ${classes.length}`)
  assert(classes.includes(0), 'should include class 0')
  assert(classes.includes(1), 'should include class 1')
  model.dispose()
})

await test('save/load roundtrip', async () => {
  const { X, y } = makeBinaryData(42, 20)
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 20, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)

  const bundleBytes = model.save()
  assert(bundleBytes instanceof Uint8Array, 'save should return Uint8Array')
  assert(bundleBytes.length > 0, 'bundle should have content')

  // Decode and check manifest
  const { manifest } = decodeBundle(bundleBytes)
  assert(manifest.typeId === 'wlearn.nn.mlp.classifier@1', `typeId = ${manifest.typeId}`)

  // Load and compare predictions
  const loaded = await MLPClassifier.load(bundleBytes)
  assert(loaded.isFitted, 'loaded model should be fitted')

  const preds1 = model.predict(X)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `prediction mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }

  model.dispose()
  loaded.dispose()
})

await test('core.load() registry dispatch', async () => {
  const { X, y } = makeBinaryData(42, 10)
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)

  const bundleBytes = model.save()
  const loaded = await load(bundleBytes)
  assert(loaded.isFitted, 'registry-loaded model should be fitted')

  model.dispose()
  loaded.dispose()
})

await test('dispose is idempotent', async () => {
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  const { X, y } = makeBinaryData(42, 10)
  model.fit(X, y)
  model.dispose()
  assert(!model.isFitted, 'should not be fitted after dispose')
  model.dispose() // should not throw
})

await test('throws after dispose', async () => {
  const model = await MLPClassifier.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  const { X, y } = makeBinaryData(42, 10)
  model.fit(X, y)
  model.dispose()
  let threw = false
  try { model.predict(X) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('capabilities', async () => {
  const model = await MLPClassifier.create()
  const caps = model.capabilities
  assert(caps.classifier === true, 'classifier should be true')
  assert(caps.regressor === false, 'regressor should be false')
  assert(caps.predictProba === true, 'predictProba should be true')
  model.dispose()
})

await test('getParams / setParams', async () => {
  const model = await MLPClassifier.create({ lr: 0.01 })
  assert(model.getParams().lr === 0.01, `expected 0.01, got ${model.getParams().lr}`)
  model.setParams({ lr: 0.001 })
  assert(model.getParams().lr === 0.001, `expected 0.001, got ${model.getParams().lr}`)
  model.dispose()
})

await test('defaultSearchSpace', async () => {
  const space = MLPClassifier.defaultSearchSpace()
  assert(space, 'search space should exist')
  assert(space.hidden_sizes, 'missing hidden_sizes')
  assert(space.lr, 'missing lr')
  assert(space.optimizer, 'missing optimizer')
})

// ═══════════════════════════════════════════════════════════════════
// MLPRegressor
// ═══════════════════════════════════════════════════════════════════

console.log('\n=== MLPRegressor ===')

await test('create() returns unfitted model', async () => {
  const model = await MLPRegressor.create()
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('throws before fit', async () => {
  const model = await MLPRegressor.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')
  model.dispose()
})

await test('regression fit and score', async () => {
  const { X, y } = makeRegressionData()
  const model = await MLPRegressor.create({
    hidden_sizes: [16], epochs: 100, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const r2 = model.score(X, y)
  assert(r2 > 0, `R2 should be > 0, got ${r2}`)
  model.dispose()
})

await test('predict returns Float64Array with correct length', async () => {
  const { X, y } = makeRegressionData(42, 20)
  const model = await MLPRegressor.create({
    hidden_sizes: [4], epochs: 10, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'should be Float64Array')
  assert(preds.length === 20, `expected 20, got ${preds.length}`)
  model.dispose()
})

await test('save/load roundtrip', async () => {
  const { X, y } = makeRegressionData(42, 20)
  const model = await MLPRegressor.create({
    hidden_sizes: [4], epochs: 20, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)

  const bundleBytes = model.save()
  const { manifest } = decodeBundle(bundleBytes)
  assert(manifest.typeId === 'wlearn.nn.mlp.regressor@1', `typeId = ${manifest.typeId}`)

  const loaded = await MLPRegressor.load(bundleBytes)
  assert(loaded.isFitted, 'loaded model should be fitted')

  const preds1 = model.predict(X)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-5, `prediction mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }

  model.dispose()
  loaded.dispose()
})

await test('core.load() registry dispatch', async () => {
  const { X, y } = makeRegressionData(42, 10)
  const model = await MLPRegressor.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  model.fit(X, y)

  const bundleBytes = model.save()
  const loaded = await load(bundleBytes)
  assert(loaded.isFitted, 'registry-loaded model should be fitted')

  model.dispose()
  loaded.dispose()
})

await test('dispose is idempotent', async () => {
  const model = await MLPRegressor.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  const { X, y } = makeRegressionData(42, 10)
  model.fit(X, y)
  model.dispose()
  assert(!model.isFitted, 'should not be fitted after dispose')
  model.dispose() // should not throw
})

await test('throws after dispose', async () => {
  const model = await MLPRegressor.create({
    hidden_sizes: [4], epochs: 5, lr: 0.01,
    optimizer: 'sgd', seed: 42
  })
  const { X, y } = makeRegressionData(42, 10)
  model.fit(X, y)
  model.dispose()
  let threw = false
  try { model.predict(X) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('capabilities', async () => {
  const model = await MLPRegressor.create()
  const caps = model.capabilities
  assert(caps.classifier === false, 'classifier should be false')
  assert(caps.regressor === true, 'regressor should be true')
  assert(caps.predictProba === false, 'predictProba should be false')
  model.dispose()
})

await test('getParams / setParams', async () => {
  const model = await MLPRegressor.create({ lr: 0.01 })
  assert(model.getParams().lr === 0.01, `expected 0.01`)
  model.setParams({ epochs: 50 })
  assert(model.getParams().epochs === 50, `expected 50`)
  model.dispose()
})

await test('defaultSearchSpace', async () => {
  const space = MLPRegressor.defaultSearchSpace()
  assert(space, 'search space should exist')
  assert(space.hidden_sizes, 'missing hidden_sizes')
  assert(space.epochs, 'missing epochs')
})

// ═══════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════

console.log(`\n${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
