// Reuse the exact training-only-scaled splits written by basis/bench/composition.py.
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import { fileURLToPath } from 'node:url'
const require = createRequire(import.meta.url)
const { BasisClassifier, BasisRegressor, BasisTransformer } = require('@wlearn/basis')
const { RFModel } = require('@wlearn/rf')
const { XGBModel } = require('@wlearn/xgboost')
const { Pipeline, accuracy, r2Score } = require('@wlearn/core')
const input = process.argv[2], output = process.argv[3]
if (!input || !output) throw new Error('Usage: node bench/basis-composition.mjs NATIVE_RUN OUTPUT')
fs.mkdirSync(output, { recursive: true })
const write = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + '\n')
const files = fs.readdirSync(input).filter(name => /-(42|43|44)\.json$/.test(name)).sort()
const maps = [['rff', 'gaussian'], ['rvfl', 'gaussian'], ['rvfl', 'swim']]
write('protocol.json', {
  files, maps, mapParams: { nComponents: 64, gamma: '1/input width', scale: 1, seed: 42 },
  readouts: { fused: 'ridge alpha=1, including one-hot classification',
    rf: '64 trees, maxFeatures=sqrt, seed=42', xgboost: '64 rounds, depth=6, eta=.1, hist, nthread=1, seed=42' },
  tuning: 'none', timing: 'one warm-process fit/predict; full Pipeline transforms included; construction/scaling/save excluded',
  memory: 'exact expanded training matrix bytes; JS heap deltas noisy; cumulative process maxRSS, not per-model peak',
  scope: 'single-threaded WASM on fixed small IID splits; no general ranking'
})
const inputs = [fileURLToPath(import.meta.url), ...['basis', 'rf', 'xgboost', 'core'].map(name => require.resolve('@wlearn/' + name))]
write('environment.json', { node: process.version, platform: process.platform, arch: process.arch,
  affinity: fs.readFileSync('/proc/self/status', 'utf8').match(/^Cpus_allowed_list:.*$/m)?.[0],
  sources: Object.fromEntries(inputs.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) })
const results = []
for (const file of files) {
  const { dataset, seed, classification, Xtrain, Xtest, ytrain, ytest } = JSON.parse(fs.readFileSync(path.join(input, file)))
  const task = classification ? 'classification' : 'regression'
  const specs = [['raw', null, 'rf'], ['raw', null, 'xgboost'], ...maps.flatMap(([method, sampling]) =>
    ['fused', 'rf', 'xgboost'].map(readout => [method, sampling, readout]))]
  for (const [method, sampling, readout] of specs) {
    const params = { method, sampling, nComponents: 64, gamma: 1 / Xtrain[0].length, seed: 42 }
    let model, map, estimator
    try {
      if (readout === 'fused') {
        model = await (classification ? BasisClassifier : BasisRegressor).create({ ...params, alpha: 1, readout: 'ridge' })
      } else {
        estimator = readout === 'rf'
          ? await RFModel.create({ task, nEstimators: 64, maxFeatures: 'sqrt', seed: 42 })
          : await XGBModel.create({ task, numRound: 64, max_depth: 6, eta: .1, tree_method: 'hist', nthread: 1, seed: 42 })
        if (method !== 'raw') map = await BasisTransformer.create({ ...params, task })
        model = map ? new Pipeline([['map', map], ['readout', estimator]]) : estimator
      }
      const heap = process.memoryUsage().heapUsed, start = performance.now()
      model.fit(Xtrain, ytrain)
      const fit_ms = performance.now() - start, before = performance.now()
      const prediction = model.predict(Xtest)
      const predict_ms = performance.now() - before
      if (prediction.some(value => !Number.isFinite(value))) throw new Error('Nonfinite prediction')
      const metric = (classification ? accuracy : r2Score)(ytest, prediction)
      results.push({ dataset, seed, task, method, sampling, readout, metric, fit_ms, predict_ms,
        materialized_training_bytes: map ? Xtrain.length * map.outputCols * 8 : 0,
        js_heap_delta_bytes: process.memoryUsage().heapUsed - heap,
        process_high_water_rss_bytes: process.resourceUsage().maxRSS * 1024,
        bundle_bytes: model.save().byteLength, predictions: Array.from(prediction) })
      write('results.json', results)
      console.log(dataset, seed, method, sampling, readout, metric.toFixed(5))
    } finally {
      if (model) model.dispose()
      else { map?.dispose(); estimator?.dispose() }
    }
  }
}
console.log(`${results.length} completed WASM composition cases`)
