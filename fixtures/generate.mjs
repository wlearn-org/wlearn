#!/usr/bin/env node

// Golden fixture generator for wlearn cross-language compatibility tests.
//
// Import modes:
//   normal:  import @wlearn/* from node_modules (npm install)
//   dev:     WLEARN_PORTS_DIR=/path/to/wlearn → import from sibling dirs
//   CI:      fixtures are committed; CI only runs verify.mjs
//
// Usage:
//   node fixtures/generate.mjs                              # normal mode
//   WLEARN_PORTS_DIR=/home/user/projects/wlearn node fixtures/generate.mjs  # dev mode

import { writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeBundle } from '@wlearn/core'
import { Pipeline } from '@wlearn/core'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PORTS_DIR = process.env.WLEARN_PORTS_DIR

// --- Import helpers ---

async function importPort(name) {
  if (PORTS_DIR) {
    // dev mode: absolute path to sibling repo
    const map = { liblinear: 'liblinear-wasm', libsvm: 'libsvm-wasm', xgboost: 'xgboost-wasm', nanoflann: 'nanoflann-wasm', ebm: 'ebm-wasm' }
    const dir = map[name]
    if (!dir) throw new Error(`Unknown port: ${name}`)
    return import(`${PORTS_DIR}/${dir}/src/index.js`)
  }
  // normal mode: node resolution
  return import(`@wlearn/${name}`)
}

// --- Deterministic LCG PRNG ---

function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

// --- Data generators ---

function makeClassificationData(rng, nSamples, nFeatures, nClasses = 2) {
  const X = []
  const y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % nClasses
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = []
  const y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

// --- Fixture writer ---

function writeFixture(name, bundle, sidecar) {
  const wlrnPath = join(__dirname, `${name}.wlrn`)
  const jsonPath = join(__dirname, `${name}.json`)
  writeFileSync(wlrnPath, bundle)
  writeFileSync(jsonPath, JSON.stringify(sidecar, null, 2))
  console.log(`  ${name}.wlrn (${bundle.length} bytes)`)
}

function makeSidecar(bundle, params, X, y, predictions, extra = {}) {
  const { manifest, toc } = decodeBundle(bundle)
  return {
    typeId: manifest.typeId,
    params,
    metadata: manifest.metadata || {},
    toc: toc.map(e => ({ id: e.id, length: e.length, sha256: e.sha256 })),
    X,
    y,
    predictions: Array.from(predictions),
    ...extra
  }
}

// --- Generate ---

async function main() {
  console.log(`Import mode: ${PORTS_DIR ? 'dev' : 'normal'}`)
  if (PORTS_DIR) console.log(`  WLEARN_PORTS_DIR=${PORTS_DIR}`)

  const { LinearModel } = await importPort('liblinear')
  const { SVMModel } = await importPort('libsvm')
  const { XGBModel } = await importPort('xgboost')

  console.log('\nGenerating fixtures...\n')

  // 1. liblinear-classifier
  {
    const rng = makeLCG(100)
    const { X, y } = makeClassificationData(rng, 50, 2)
    const params = { solver: 0, C: 1.0, eps: 0.01 }
    const model = await LinearModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('liblinear-classifier', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 2. libsvm-classifier
  {
    const rng = makeLCG(200)
    const { X, y } = makeClassificationData(rng, 50, 2)
    const params = { svmType: 0, kernel: 2, C: 10, gamma: 0.5 }
    const model = await SVMModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('libsvm-classifier', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 3. xgboost-binary
  {
    const rng = makeLCG(300)
    const { X, y } = makeClassificationData(rng, 50, 3)
    const params = { objective: 'binary:logistic', max_depth: 3, eta: 0.3, numRound: 20 }
    const model = await XGBModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('xgboost-binary', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 4. xgboost-multiclass
  {
    const rng = makeLCG(400)
    const { X, y } = makeClassificationData(rng, 75, 3, 3)
    const params = { objective: 'multi:softprob', num_class: 3, max_depth: 3, eta: 0.3, numRound: 20 }
    const model = await XGBModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('xgboost-multiclass', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 5. xgboost-regressor
  {
    const rng = makeLCG(500)
    const { X, y } = makeRegressionData(rng, 50, 2)
    const params = { objective: 'reg:squarederror', max_depth: 3, eta: 0.3, numRound: 20 }
    const model = await XGBModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('xgboost-regressor', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 6. pipeline-single (single-step pipeline wrapping xgboost)
  {
    const rng = makeLCG(600)
    const { X, y } = makeClassificationData(rng, 50, 3)
    const params = { objective: 'binary:logistic', max_depth: 3, eta: 0.3, numRound: 20 }
    const model = await XGBModel.create(params)
    const pipe = new Pipeline([['classifier', model]])
    pipe.fit(X, y)
    const preds = pipe.predict(X)
    const bundle = pipe.save()
    writeFixture('pipeline-single', bundle,
      makeSidecar(bundle, pipe.getParams(), X, y, preds))
    pipe.dispose()
  }

  // 7. nanoflann-classifier
  {
    const { KNNModel } = await importPort('nanoflann')
    const rng = makeLCG(700)
    const { X, y } = makeClassificationData(rng, 50, 2)
    const params = { k: 5, metric: 'l2', leafMaxSize: 10, task: 'classification' }
    const model = await KNNModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('nanoflann-classifier', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 8. nanoflann-regressor
  {
    const { KNNModel } = await importPort('nanoflann')
    const rng = makeLCG(800)
    const { X, y } = makeRegressionData(rng, 50, 2)
    const params = { k: 5, metric: 'l2', leafMaxSize: 10, task: 'regression' }
    const model = await KNNModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('nanoflann-regressor', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 9. ebm-classifier
  {
    const { EBMModel } = await importPort('ebm')
    const rng = makeLCG(900)
    const { X, y } = makeClassificationData(rng, 100, 2)
    const params = { maxRounds: 100, earlyStoppingRounds: 20, maxInteractions: 0, seed: 42 }
    const model = await EBMModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('ebm-classifier', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  // 10. ebm-regressor
  {
    const { EBMModel } = await importPort('ebm')
    const rng = makeLCG(1000)
    const { X, y } = makeRegressionData(rng, 100, 2)
    const params = { objective: 'regression', maxRounds: 100, earlyStoppingRounds: 20, maxInteractions: 0, seed: 42 }
    const model = await EBMModel.create(params)
    model.fit(X, y)
    const preds = model.predict(X)
    const bundle = model.save()
    writeFixture('ebm-regressor', bundle,
      makeSidecar(bundle, model.getParams(), X, y, preds))
    model.dispose()
  }

  console.log('\nDone. All fixtures generated.')
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
