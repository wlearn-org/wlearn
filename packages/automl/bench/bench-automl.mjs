/**
 * AutoML benchmark: wlearn JS on Friedman 1-3, moons, hastie.
 * Sizes: 500, 2000, 10000.
 *
 * Uses WASM model packages from sibling repos (../../../<lib>-wasm/).
 * Skips any model that can't be imported.
 */

import { autoFit } from '../src/auto-fit.js'
import { makeFriedman1, makeFriedman2, makeFriedman3, makeMoons, makeHastie, trainTestSplit } from './datasets.mjs'

// --- Dynamic model imports (graceful skip) ---

async function loadModels() {
  const models = {}
  const base = new URL('../../../../', import.meta.url).pathname

  const tryImport = async (name, path, cls) => {
    try {
      const mod = await import(path)
      models[name] = mod[cls]
      console.log(`  ${name}: loaded`)
    } catch (e) {
      console.log(`  ${name}: skipped (${e.message.split('\n')[0]})`)
    }
  }

  await tryImport('xgb', base + 'xgboost-wasm/src/index.js', 'XGBModel')
  await tryImport('linear', base + 'liblinear-wasm/src/index.js', 'LinearModel')
  await tryImport('svm', base + 'libsvm-wasm/src/index.js', 'SVMModel')
  await tryImport('knn', base + 'nanoflann-wasm/src/index.js', 'KNNModel')
  await tryImport('ebm', base + 'ebm-wasm/src/index.js', 'EBMModel')
  await tryImport('lgb', base + 'lightgbm-wasm/src/index.js', 'LGBModel')
  await tryImport('tsetlin', base + 'tsetlin-wasm/src/index.js', 'TsetlinModel')

  return models
}

function makeRegSpecs(modelMap) {
  const specs = []
  if (modelMap.xgb) specs.push({ name: 'xgb', cls: modelMap.xgb, params: { objective: 'reg:squarederror', numRound: 100 } })
  if (modelMap.linear) specs.push({ name: 'linear', cls: modelMap.linear, params: { solver: 11, C: 1.0 } })
  if (modelMap.svm) specs.push({ name: 'svm', cls: modelMap.svm, params: { svmType: 3, kernel: 2, C: 1.0, gamma: 0 } })
  if (modelMap.knn) specs.push({ name: 'knn', cls: modelMap.knn, params: { k: 5, task: 'regression' } })
  if (modelMap.ebm) specs.push({ name: 'ebm', cls: modelMap.ebm, params: { objective: 'regression' } })
  if (modelMap.lgb) specs.push({ name: 'lgb', cls: modelMap.lgb, params: { objective: 'regression', numRound: 100, verbosity: -1 } })
  return specs
}

function makeClsSpecs(modelMap) {
  const specs = []
  if (modelMap.xgb) specs.push({ name: 'xgb', cls: modelMap.xgb, params: { objective: 'multi:softprob', numRound: 100 } })
  if (modelMap.linear) specs.push({ name: 'linear', cls: modelMap.linear, params: { solver: 0, C: 1.0 } })
  if (modelMap.svm) specs.push({ name: 'svm', cls: modelMap.svm, params: { svmType: 0, kernel: 2, C: 1.0, gamma: 0, probability: 1 } })
  if (modelMap.knn) specs.push({ name: 'knn', cls: modelMap.knn, params: { k: 5, task: 'classification' } })
  if (modelMap.ebm) specs.push({ name: 'ebm', cls: modelMap.ebm, params: { objective: 'classification' } })
  if (modelMap.lgb) specs.push({ name: 'lgb', cls: modelMap.lgb, params: { objective: 'binary', numRound: 100, verbosity: -1 } })
  return specs
}

// --- Scoring ---

function r2Score(yTrue, yPred) {
  const n = yTrue.length
  let mean = 0
  for (let i = 0; i < n; i++) mean += yTrue[i]
  mean /= n
  let ssTot = 0, ssRes = 0
  for (let i = 0; i < n; i++) {
    ssTot += (yTrue[i] - mean) ** 2
    ssRes += (yTrue[i] - yPred[i]) ** 2
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot
}

function accuracy(yTrue, yPred) {
  let correct = 0
  for (let i = 0; i < yTrue.length; i++) {
    if (Math.round(yPred[i]) === yTrue[i]) correct++
  }
  return correct / yTrue.length
}

// --- Run single benchmark ---

async function runWlearn(specs, Xtrain, ytrain, Xtest, ytest, task, strategy) {
  if (specs.length === 0) return { score: null, time: 0 }

  const t0 = performance.now()
  try {
    const result = await autoFit(specs, Xtrain, ytrain, {
      strategy,
      cv: 5,
      seed: 42,
      task,
      nIter: 20,
    })
    const elapsed = (performance.now() - t0) / 1000

    if (!result.model) return { score: null, time: elapsed }

    const preds = result.model.predict(Xtest)
    const score = task === 'classification'
      ? accuracy(ytest, preds)
      : r2Score(ytest, preds)

    result.model.dispose()
    return { score, time: elapsed }
  } catch (e) {
    const elapsed = (performance.now() - t0) / 1000
    console.error(`  Error (${strategy}): ${e.message}`)
    return { score: null, time: elapsed }
  }
}

// --- Format helpers ---

function fmtScore(s) {
  if (s === null) return '  --  '
  return s.toFixed(4).padStart(6)
}

function fmtTime(t) {
  if (t === 0) return '  -- '
  if (t < 1) return `${(t * 1000).toFixed(0).padStart(4)}ms`
  return `${t.toFixed(1).padStart(5)}s`
}

// --- Main ---

async function main() {
  console.log()
  console.log('AutoML Benchmark: wlearn JS')
  console.log('Loading models...')

  const modelMap = await loadModels()
  const nModels = Object.keys(modelMap).length
  console.log(`${nModels} model(s) available`)
  console.log()

  if (nModels === 0) {
    console.log('No WASM model packages found. Link them with npm or adjust import paths.')
    process.exit(1)
  }

  const sizes = [500, 2000]
  const datasets = {
    friedman1: (n, seed) => ({ ...makeFriedman1(n, { seed }), task: 'regression' }),
    friedman2: (n, seed) => ({ ...makeFriedman2(n, { seed }), task: 'regression' }),
    friedman3: (n, seed) => ({ ...makeFriedman3(n, { seed }), task: 'regression' }),
    moons:     (n, seed) => ({ ...makeMoons(n, { seed }), task: 'classification' }),
    hastie:    (n, seed) => ({ ...makeHastie(n, { seed }), task: 'classification' }),
  }

  // Header
  console.log('| Dataset   |     n | wlearn-portfolio |  time | wlearn-random    |  time |')
  console.log('|-----------|------:|-----------------:|------:|-----------------:|------:|')

  for (const n of sizes) {
    for (const [name, gen] of Object.entries(datasets)) {
      const { X, y, task } = gen(n, 42)
      const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, { testSize: 0.2, seed: 42 })

      const specs = task === 'classification' ? makeClsSpecs(modelMap) : makeRegSpecs(modelMap)

      const portfolio = await runWlearn(specs, Xtrain, ytrain, Xtest, ytest, task, 'portfolio')
      const random = await runWlearn(specs, Xtrain, ytrain, Xtest, ytest, task, 'random')

      console.log(
        `| ${name.padEnd(9)} | ${String(n).padStart(5)} |` +
        ` ${fmtScore(portfolio.score).padStart(16)} | ${fmtTime(portfolio.time).padStart(5)} |` +
        ` ${fmtScore(random.score).padStart(16)} | ${fmtTime(random.time).padStart(5)} |`
      )
    }
  }

  console.log()
}

main().catch(e => { console.error(e); process.exit(1) })
