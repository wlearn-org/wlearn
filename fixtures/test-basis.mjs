import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { test } = require('node:test')
const assert = require('node:assert/strict')
const core = require('@wlearn/core')
const { BasisTransformer, BasisClassifier, BasisRegressor } = require('@wlearn/basis')
const { autoFit, sampleConfig } = require('@wlearn/automl')
const X = Array.from({ length: 30 }, (_, i) => [(i - 15) / 8, Math.sin(i)])
const y = X.map(row => 1 + 2 * row[0])
const labels = X.map(row => row[0] < 0 ? -9 : 27)

test('core Pipeline and ensemble use basis through the common contract', async () => {
  const { VotingEnsemble } = require('@wlearn/ensemble')
  const pipe = new core.Pipeline([['basis', await BasisClassifier.create({ nComponents: 12 })]])
  const vote = await VotingEnsemble.create({ estimators: [
    ['rff', BasisClassifier, { method: 'rff', nComponents: 12 }],
    ['rvfl', BasisClassifier, { method: 'rvfl', nComponents: 12 }]
  ] })
  let loaded
  try {
    pipe.fit(X, labels)
    assert.deepEqual(pipe.classes, Int32Array.of(-9, 27))
    loaded = await core.load(pipe.save())
    assert.deepEqual(await loaded.predict(X), pipe.predict(X))
    loaded.dispose()
    await vote.fit(X, labels)
    loaded = await core.load(vote.save())
    assert.deepEqual(await loaded.predictProba(X), vote.predictProba(X))
    assert.ok(vote.score(X, labels) > 0.9)
  } finally { pipe.dispose(); vote.dispose(); loaded?.dispose() }
})

test('AutoML searches basis without model-specific engine changes', async () => {
  const { autoFit } = require('@wlearn/automl')
  const result = await autoFit([{ name: 'basis', cls: BasisRegressor,
    searchSpace: { method: { type: 'categorical', values: ['rff', 'rvfl'] }, nComponents: { type: 'categorical', values: [8] } }
  }], X, y, { nIter: 2, cv: 2, task: 'regression', seed: 42 })
  try {
    assert.ok(result.model.isFitted)
    assert.equal(result.archive.records({ status: 'ok' }).length, 2)
    assert.ok(Number.isFinite(result.model.score(X, y)))
  } finally { result.model?.dispose() }
})

test('basis default search uses public conditional sampling', () => {
  const rng = core.makeLCG(42)
  for (let i = 0; i < 60; i++) {
    const config = sampleConfig(BasisRegressor.defaultSearchSpace(), rng)
    assert.equal(Object.hasOwn(config, 'sampling'), config.method === 'elm')
  }
})

for (const [name, packageName, exportName, params] of [
  ['GAM', '@wlearn/gam', 'GAMModel', { family: 'gaussian', penalty: 'ridge', nLambda: 8 }],
  ['RF', '@wlearn/rf', 'RFModel', { task: 'regression', nEstimators: 12 }],
  ['XGBoost', '@wlearn/xgboost', 'XGBModel', { task: 'regression', numRound: 8 }]
]) test(`basis Transformer -> ${name} fits and reloads through core`, async () => {
  const Class = require(packageName)[exportName]
  const pipe = new core.Pipeline([
    ['map', await BasisTransformer.create({ method: 'elm', sampling: 'swim', nComponents: 8 })],
    ['model', await Class.create(params)]
  ])
  let restored
  try {
    await pipe.fit(X, y)
    const expected = await pipe.predict(X)
    assert.equal(expected.length, X.length)
    assert(expected.every(Number.isFinite))
    restored = await core.load(pipe.save())
    assert.deepEqual(await restored.predict(X), expected)
  } finally { pipe.dispose(); restored?.dispose() }
})

test('AutoML fits supervised basis maps inside each training fold', async () => {
  const fits = []
  class MapReadout {
    static classId = 'test.basis-map-readout'
    static defaultSearchSpace() { return { nComponents: { type: 'categorical', values: [6, 8] } } }
    static async create(params) {
      const map = await BasisTransformer.create({ ...params, method: 'elm', sampling: 'swim' })
      const fitTransform = map.fitTransform.bind(map)
      map.fitTransform = (foldX, foldY) => { fits.push({ rows: foldX.rows, targets: foldY.length }); return fitTransform(foldX, foldY) }
      return new core.Pipeline([['map', map], ['readout', await BasisRegressor.create({ method: 'rff', nComponents: 8 })]])
    }
  }
  const result = await autoFit([{ name: 'map-readout', cls: MapReadout }], X, y, { nIter: 2, cv: 3, task: 'regression', seed: 42, ensemble: false })
  try {
    assert.equal(result.archive.records({ status: 'ok' }).length, 2)
    assert.equal(fits.filter(f => f.rows === 20 && f.targets === 20).length, 6)
    assert.equal(fits.filter(f => f.rows === 30 && f.targets === 30).length, 1)
  } finally { result.model?.dispose() }
})
