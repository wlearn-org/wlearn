import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const core = require('../js/core/src/index.js')
const workspace = resolve(
  process.env.WLEARN_PORTS_DIR || join(dirname(fileURLToPath(import.meta.url)), '..', '..')
)

const binaryY = new Int32Array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
const integerRegressionY = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
const X = {
  data: new Float64Array(Array.from(
    { length: 24 },
    (_, i) => i % 2 === 0 ? Math.floor(i / 2) / 11 : ((Math.floor(i / 2) * 5) % 12) / 11
  )),
  rows: 12,
  cols: 2,
}

const specs = [
  {
    name: 'liblinear', repo: 'liblinear-wasm', module: 'src/model.js', export: 'LinearModel',
    wasm: 'src/wasm.js', load: 'loadLinear', params: { C: 1 },
  },
  {
    name: 'libsvm', repo: 'libsvm-wasm', module: 'src/model.js', export: 'SVMModel',
    wasm: 'src/wasm.js', load: 'loadSVM', params: { kernel: 'LINEAR', C: 1 },
  },
  {
    name: 'xgboost', repo: 'xgboost-wasm', module: 'src/model.js', export: 'XGBModel',
    wasm: 'src/wasm.js', load: 'loadXGB', params: { numRound: 3, max_depth: 2 },
  },
  {
    name: 'lightgbm', repo: 'lightgbm-wasm', module: 'src/model.js', export: 'LGBModel',
    wasm: 'src/wasm.js', load: 'loadLGB',
    params: { numRound: 3, max_depth: 2, min_data_in_leaf: 1, verbosity: -1 },
  },
  {
    name: 'nanoflann', repo: 'nanoflann-wasm', module: 'src/model.js', export: 'KNNModel',
    wasm: 'src/wasm.js', load: 'loadNanoflann', params: { k: 3 },
  },
  {
    name: 'ebm', repo: 'ebm-wasm', module: 'src/model.js', export: 'EBMModel',
    wasm: 'src/wasm.js', load: 'loadEBM',
    params: { maxRounds: 10, maxInteractions: 0, minSamplesLeaf: 1 },
  },
  {
    name: 'stochtree', repo: 'stochtree-wasm', module: 'src/model.js', export: 'BARTModel',
    wasm: 'src/wasm.js', load: 'loadStochtree',
    params: { numTrees: 5, numGfr: 1, numBurnin: 2, numSamples: 2, minSamplesLeaf: 1 },
  },
  {
    name: 'tsetlin', repo: 'tsetlin-wasm', module: 'src/model.js', export: 'TsetlinModel',
    wasm: 'src/wasm.js', load: 'loadTsetlin',
    params: { nClauses: 20, epochs: 3, threshold: 10, s: 3 },
  },
  {
    name: 'rf', repo: 'rf', module: 'js/src/model.js', export: 'RFModel',
    wasm: 'js/src/wasm.js', load: 'loadRF', params: { nTrees: 5, maxDepth: 3, minSamplesLeaf: 1 },
  },
  {
    name: 'gam', repo: 'gam', module: 'js/src/model.js', export: 'GAMModel',
    wasm: 'js/src/wasm.js', load: 'loadGAM', params: { maxIter: 50, lambda: 0.01 },
  },
]

function modelClass(spec) {
  const implementation = require(join(workspace, spec.repo, spec.module))[spec.export]
  const load = require(join(workspace, spec.repo, spec.wasm))[spec.load]
  return core.createModelClass(implementation, implementation, {
    name: `${spec.export}Unified`,
    load,
  })
}

async function verifyCase(Model, spec, task, y) {
  let model
  try {
    model = await Model.create({ ...spec.params, ...(task ? { task } : {}) })
    model.fit(X, y)
    const manifest = core.decodeBundle(model.save()).manifest
    const expectedTask = task || 'classification'

    assert.equal(model.task, expectedTask, `${spec.name}: wrapper task`)
    const typeVersion = spec.typeVersions?.[expectedTask] || spec.typeVersion || 1
    assert.equal(
      manifest.typeId,
      `${spec.typePrefix || `wlearn.${spec.name}`}.${expectedTask === 'classification' ? 'classifier' : 'regressor'}@${typeVersion}`,
      `${spec.name}: bundle typeId`
    )
    assert.equal(manifest.params.task, expectedTask, `${spec.name}: persisted task`)
    assert.equal(model.capabilities.classifier, expectedTask === 'classification', `${spec.name}: classifier capability`)
    assert.equal(model.capabilities.regressor, expectedTask === 'regression', `${spec.name}: regressor capability`)
  } finally {
    if (model) model.dispose()
  }
}

let passed = 0
for (const spec of specs) {
  const Model = modelClass(spec)
  await verifyCase(Model, spec, null, binaryY)
  await verifyCase(Model, spec, 'regression', integerRegressionY)
  passed += 2
  console.log(`PASS ${spec.name}: inferred classification + explicit integer regression`)
}

const xlearn = require(join(workspace, 'xlearn-wasm', 'src/lr.js'))
const loadXLearn = require(join(workspace, 'xlearn-wasm', 'src/wasm.js')).loadXLearn
const xlearnSpec = {
  name: 'xlearn-lr',
  typePrefix: 'wlearn.xlearn.lr',
  typeVersions: { classification: 2, regression: 1 },
  params: { epoch: 5 },
}
const XLearnLR = core.createModelClass(
  xlearn.XLearnLRClassifier,
  xlearn.XLearnLRRegressor,
  { name: 'XLearnLR', load: loadXLearn }
)
await verifyCase(XLearnLR, xlearnSpec, null, binaryY)
await verifyCase(XLearnLR, xlearnSpec, 'regression', integerRegressionY)
passed += 2
console.log('PASS xlearn-lr: deferred task-specific factories')

console.log(`\nUnified model task conformance: ${passed}/${specs.length * 2 + 2} cases passed`)
