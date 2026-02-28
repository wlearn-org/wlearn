/**
 * Cross-language test: Train in JS, save bundle, verify Python can load
 * and produce identical predictions.
 *
 * Usage: node test/test_cross_lang.js
 * Writes a fixture bundle to fixtures/ for Python verification.
 */

import { writeFileSync, mkdirSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

const { MLPClassifier, MLPRegressor } = await import('../src/index.js')

// ── Deterministic data (must match Python test) ──────────────────

function makeBinaryData(n = 30) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x0 = (i * 7 % 31 - 15) / 15.0
    const x1 = (i * 13 % 31 - 15) / 15.0
    X.push([x0, x1])
    y.push(x0 + x1 > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeRegressionData(n = 30) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x0 = (i * 7 % 31 - 15) / 15.0
    const x1 = (i * 13 % 31 - 15) / 15.0
    X.push([x0, x1])
    y.push(2.0 * x0 + 3.0 * x1 + 0.5)
  }
  return { X, y }
}

// ── Generate fixtures ──────────────────────────────────────────────

const fixturesDir = join(__dirname, '..', '..', '..', 'fixtures', 'nn')
mkdirSync(fixturesDir, { recursive: true })

// Classifier
const { X: cX, y: cY } = makeBinaryData()
const clf = await MLPClassifier.create({
  hidden_sizes: [4], epochs: 20, lr: 0.01,
  optimizer: 'sgd', seed: 42
})
clf.fit(cX, cY)
const clfPreds = clf.predict(cX)
const clfProba = clf.predictProba(cX)
const clfBundle = clf.save()

writeFileSync(join(fixturesDir, 'mlp_classifier.wlrn'), clfBundle)
writeFileSync(join(fixturesDir, 'mlp_classifier.json'), JSON.stringify({
  typeId: 'wlearn.nn.mlp.classifier@1',
  X: cX,
  predictions: [...clfPreds],
  probabilities: [...clfProba],
  score: clf.score(cX, cY),
  nrClass: clf.nrClass,
  classes: clf.classes
}, null, 2))

console.log(`Classifier bundle: ${clfBundle.length} bytes, score=${clf.score(cX, cY).toFixed(4)}`)
clf.dispose()

// Regressor
const { X: rX, y: rY } = makeRegressionData()
const reg = await MLPRegressor.create({
  hidden_sizes: [4], epochs: 20, lr: 0.01,
  optimizer: 'sgd', seed: 42
})
reg.fit(rX, rY)
const regPreds = reg.predict(rX)
const regBundle = reg.save()

writeFileSync(join(fixturesDir, 'mlp_regressor.wlrn'), regBundle)
writeFileSync(join(fixturesDir, 'mlp_regressor.json'), JSON.stringify({
  typeId: 'wlearn.nn.mlp.regressor@1',
  X: rX,
  predictions: [...regPreds],
  score: reg.score(rX, rY)
}, null, 2))

console.log(`Regressor bundle: ${regBundle.length} bytes, R2=${reg.score(rX, rY).toFixed(4)}`)
reg.dispose()

console.log(`\nFixtures written to ${fixturesDir}`)
