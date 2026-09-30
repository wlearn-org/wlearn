import { createRequire } from 'node:module'
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import assert from 'node:assert/strict'
const require = createRequire(import.meta.url)
const { RFModel } = require('@wlearn/rf')
const { load } = require('@wlearn/core')
const directory = mkdtempSync(join(tmpdir(), 'wlearn-rf-labels-'))
const X = Array.from({ length: 30 }, (_, i) => [i % 3, i / 30])
const y = X.map(row => [-9, 4, 200][row[0]])
const model = await RFModel.create({ nTrees: 8, seed: 42, classWeight: 'balanced' })
try {
  model.fit(X, y)
  writeFileSync(join(directory, 'js.wlrn'), model.save())
  writeFileSync(join(directory, 'data.json'), JSON.stringify({ X, y, predictions: [...model.predict(X)], probabilities: [...model.predictProba(X)] }))
  const program = `
import json, pathlib, sys, numpy as np
from wlearn import load
from wlearn_rf import RFModel
path = pathlib.Path(sys.argv[1])
d = json.loads((path/'data.json').read_text())
m = load(path/'js.wlrn')
try:
    np.testing.assert_array_equal(m.classes, [-9, 4, 200])
    np.testing.assert_array_equal(m.predict(d['X']), d['predictions'])
    np.testing.assert_allclose(m.predict_proba(d['X']).ravel(), d['probabilities'], atol=1e-12, rtol=0)
    m.save(path/'roundtrip.wlrn')
finally: m.dispose()
m = RFModel({'n_estimators': 8, 'seed': 42, 'class_weight': 'balanced'})
try:
    m.fit(d['X'], d['y'])
    m.save(path/'python.wlrn')
    (path/'python.json').write_text(json.dumps({'predictions': m.predict(d['X']).tolist(), 'probabilities': m.predict_proba(d['X']).ravel().tolist()}))
finally: m.dispose()
`
  const result = spawnSync(process.env.WLEARN_PYTHON || 'python3', ['-c', program, directory], { encoding: 'utf8' })
  assert.equal(result.status, 0, result.stderr || result.stdout)
  for (const filename of ['roundtrip', 'python']) {
    const restored = await load(readFileSync(join(directory, filename + '.wlrn')))
    try {
      const expected = JSON.parse(readFileSync(join(directory, filename === 'python' ? 'python.json' : 'data.json'), 'utf8'))
      assert.deepEqual([...restored.classes], [-9, 4, 200])
      assert.deepEqual([...restored.predict(X)], expected.predictions)
      const actual = restored.predictProba(X)
      assert(actual.every((v, i) => Math.abs(v - expected.probabilities[i]) <= 1e-12))
    } finally { restored.dispose() }
  }
  console.log('RF classifier@2: JS -> Python -> JS and Python -> JS passed')
} finally {
  model.dispose()
  rmSync(directory, { recursive: true, force: true })
}
