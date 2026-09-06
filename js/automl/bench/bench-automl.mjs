/**
 * AutoML benchmark: wlearn JS on Friedman 1-3, moons, hastie.
 * Sizes: 500, 2000; fixed seeds: 42, 43, 44.
 *
 * Uses WASM model packages from sibling repos (../../../<lib>-wasm/).
 * Every selected model is required. Use --smoke for the bounded CI profile.
 */

import { autoFit } from '../src/auto-fit.js'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { readFileSync, writeFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { cpus } from 'node:os'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { accuracy, r2Score } from '@wlearn/core'
import { makeFriedman1, makeFriedman2, makeFriedman3, makeMoons, makeHastie, trainTestSplit } from './datasets.mjs'

const MODEL_PACKAGES = {
  xgb: ['xgboost-wasm', 'XGBModel'], linear: ['liblinear-wasm', 'LinearModel'],
  svm: ['libsvm-wasm', 'SVMModel'], knn: ['nanoflann-wasm', 'KNNModel'],
  ebm: ['ebm-wasm', 'EBMModel'], lgb: ['lightgbm-wasm', 'LGBModel'],
}

function revision(root) {
  const result = spawnSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' })
  return result.status === 0 ? result.stdout.trim() : null
}

function sourceHash(root) {
  const hash = createHash('sha256')
  function visit(dir, prefix = '') {
    for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const name = `${prefix}${entry.name}`
      if (entry.isDirectory()) visit(join(dir, entry.name), `${name}/`)
      else if (entry.isFile()) hash.update(name).update('\0').update(readFileSync(join(dir, entry.name))).update('\0')
    }
  }
  visit(root)
  return hash.digest('hex')
}

async function loadModels(names) {
  const models = {}, sources = {}
  const base = fileURLToPath(new URL('../../../../', import.meta.url))
  for (const name of names) {
    if (!Object.hasOwn(MODEL_PACKAGES, name)) throw new Error(`Unknown required model: ${name}`)
    const [repo, cls] = MODEL_PACKAGES[name]
    const root = join(base, repo)
    const mod = await import(pathToFileURL(join(root, 'src/index.js')).href)
    if (typeof mod[cls]?.create !== 'function') throw new Error(`Missing required model constructor: ${name}`)
    models[name] = mod[cls]
    const artifacts = {}
    const wasmDir = join(root, 'wasm')
    for (const file of readdirSync(wasmDir).filter(file => /\.(js|wasm)$/.test(file)).sort()) {
      const bytes = readFileSync(join(wasmDir, file))
      artifacts[file] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') }
    }
    sources[name] = {
      revision: revision(root),
      version: JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).version,
      src_sha256: sourceHash(join(root, 'src')),
      wasm: artifacts,
    }
  }
  return { models, sources }
}

function makeRegSpecs(modelMap) {
  const specs = []
  if (modelMap.xgb) specs.push({ name: 'xgb', cls: modelMap.xgb, params: { objective: 'reg:squarederror', numRound: 100 } })
  if (modelMap.linear) specs.push({ name: 'linear', cls: modelMap.linear, params: { solver: 11, C: 1.0 } })
  if (modelMap.svm) specs.push({ name: 'svm', cls: modelMap.svm, params: { svmType: 3, kernel: 2, C: 1.0, gamma: 0 } })
  if (modelMap.knn) specs.push({ name: 'knn', cls: modelMap.knn, params: { k: 5, task: 'regression' } })
  if (modelMap.ebm) specs.push({ name: 'ebm', cls: modelMap.ebm, params: { objective: 'regression' } })
  if (modelMap.lgb) specs.push({ name: 'lgb', cls: modelMap.lgb, params: { objective: 'regression', numRound: 100, verbosity: -1 } })
  return specs.map(spec => ({ ...spec, classId: `wlearn.bench.${spec.cls.name}` }))
}

function makeClsSpecs(modelMap) {
  const specs = []
  if (modelMap.xgb) specs.push({ name: 'xgb', cls: modelMap.xgb, params: { objective: 'multi:softprob', numRound: 100 } })
  if (modelMap.linear) specs.push({ name: 'linear', cls: modelMap.linear, params: { solver: 0, C: 1.0 } })
  if (modelMap.svm) specs.push({ name: 'svm', cls: modelMap.svm, params: { svmType: 0, kernel: 2, C: 1.0, gamma: 0, probability: 1 } })
  if (modelMap.knn) specs.push({ name: 'knn', cls: modelMap.knn, params: { k: 5, task: 'classification' } })
  if (modelMap.ebm) specs.push({ name: 'ebm', cls: modelMap.ebm, params: { objective: 'classification' } })
  if (modelMap.lgb) specs.push({ name: 'lgb', cls: modelMap.lgb, params: { objective: 'binary', numRound: 100, verbosity: -1 } })
  return specs.map(spec => ({ ...spec, classId: `wlearn.bench.${spec.cls.name}` }))
}

// --- Run single benchmark ---

export async function runWlearn(specs, Xtrain, ytrain, Xtest, ytest, task, strategy, {
  autoFitFn = autoFit, seed = 42, cv = 5, nIter = 20,
} = {}) {
  if (specs.length === 0) throw new Error('Benchmark requires at least one model candidate')
  const heapBefore = process.memoryUsage().heapUsed
  const t0 = performance.now()
  let model
  try {
    const result = await autoFitFn(specs, Xtrain, ytrain, { strategy, cv, seed, task, nIter })
    model = result.model
    const fitMs = performance.now() - t0
    if (!model) throw new Error('AutoML did not return a fitted model')
    const failures = result.archive?.records({ status: 'failed' }) || []
    if (failures.length) {
      throw new Error(`Benchmark candidate failures: ${JSON.stringify(failures.map(record => ({
        model: record.metadata?.modelName, error: record.error,
      })))}`)
    }
    const predictStart = performance.now()
    const preds = await model.predict(Xtest)
    const predictMs = performance.now() - predictStart
    const score = task === 'classification' ? accuracy(ytest, preds) : r2Score(ytest, preds)
    if (!Number.isFinite(score)) throw new Error('Benchmark score is not finite')
    const bytes = model.save()
    if (!(bytes instanceof Uint8Array) || !bytes.byteLength) throw new Error('Benchmark model returned an empty or invalid bundle')
    return {
      score, fit_ms: fitMs, predict_ms: predictMs, bundle_bytes: bytes.byteLength,
      js_heap_delta_bytes: process.memoryUsage().heapUsed - heapBefore,
      // Backend heaps are private; report unknown rather than inventing a peak.
      peak_wasm_memory_bytes: null,
      best_model: result.bestModelName || null,
    }
  } finally {
    model?.dispose()
  }
}

async function main() {
  const options = { smoke: false, models: Object.keys(MODEL_PACKAGES), output: null }
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i]
    if (arg === '--smoke') options.smoke = true
    else if (arg === '--output' && process.argv[i + 1]) options.output = process.argv[++i]
    else if (arg === '--models' && process.argv[i + 1]) options.models = process.argv[++i].split(',')
    else throw new Error(`Unknown or incomplete benchmark option: ${arg}`)
  }
  if (!options.models.length || new Set(options.models).size !== options.models.length) throw new Error('Select unique required models')
  const root = fileURLToPath(new URL('../../../', import.meta.url))
  const report = {
    schema_version: 1, status: 'ok', profile: options.smoke ? 'smoke' : 'full',
    environment: { node: process.version, platform: process.platform, arch: process.arch, cpu: cpus()[0]?.model || null },
    source_revision: revision(root), required_models: options.models, models: {}, results: [],
    automl_src_sha256: sourceHash(fileURLToPath(new URL('../src/', import.meta.url))),
    benchmark_sha256: sourceHash(fileURLToPath(new URL('./', import.meta.url))),
  }
  const datasets = {
    friedman1: [makeFriedman1, 'regression'], friedman2: [makeFriedman2, 'regression'],
    friedman3: [makeFriedman3, 'regression'], moons: [makeMoons, 'classification'],
    hastie: [makeHastie, 'classification'],
  }
  try {
    const { models, sources } = await loadModels(options.models)
    report.models = sources
    const names = options.smoke ? ['friedman1', 'moons'] : Object.keys(datasets)
    for (const seed of (options.smoke ? [42] : [42, 43, 44])) {
      for (const n of (options.smoke ? [128] : [500, 2000])) {
        for (const name of names) {
          const [generate, task] = datasets[name]
          const { X, y } = generate(n, { seed })
          const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, { testSize: 0.2, seed })
          const specs = task === 'classification' ? makeClsSpecs(models) : makeRegSpecs(models)
          for (const strategy of ['portfolio', 'random']) {
            const metrics = await runWlearn(specs, Xtrain, ytrain, Xtest, ytest, task, strategy, {
              seed, cv: options.smoke ? 2 : 5, nIter: options.smoke ? 2 : 20,
            })
            report.results.push({ dataset: name, rows: n, seed, task, strategy, ...metrics })
          }
        }
      }
    }
  } catch (error) {
    report.status = 'failed'
    report.error = error.message
    process.exitCode = 1
  }
  const json = JSON.stringify(report, null, 2) + '\n'
  if (options.output) writeFileSync(options.output, json)
  else process.stdout.write(json)
  if (report.status !== 'ok') console.error(report.error)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(e => { console.error(e); process.exitCode = 1 })
}
