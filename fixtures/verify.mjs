#!/usr/bin/env node

// Validate the committed JS-produced golden corpus and execute every bundle.

import assert from 'node:assert/strict'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeBundle, load, validateBundle } from '@wlearn/core'
import { assertPredictionParity } from './verify-utils.mjs'

const fixturesDir = dirname(fileURLToPath(import.meta.url))
const portsDir = process.env.WLEARN_PORTS_DIR
const expected = [
  'ebm-classifier',
  'ebm-regressor',
  'liblinear-classifier',
  'libsvm-classifier',
  'lightgbm-binary',
  'lightgbm-multiclass',
  'lightgbm-regressor',
  'nanoflann-classifier',
  'nanoflann-regressor',
  'pipeline-preprocess-liblinear',
  'pipeline-single',
  'preprocess-tabular',
  'stochtree-classifier',
  'stochtree-regressor',
  'xgboost-binary',
  'xgboost-multiclass',
  'xgboost-regressor',
  'xlearn-classifier',
  'xlearn-regressor',
]

async function importModels() {
  const models = [
    ['liblinear', 'liblinear-wasm'],
    ['libsvm', 'libsvm-wasm'],
    ['xgboost', 'xgboost-wasm'],
    ['nanoflann', 'nanoflann-wasm'],
    ['ebm', 'ebm-wasm'],
    ['lightgbm', 'lightgbm-wasm'],
    ['stochtree', 'stochtree-wasm'],
    ['xlearn', 'xlearn-wasm'],
  ]
  for (const [pkg, repo] of models) {
    await import(portsDir ? `${portsDir}/${repo}/src/index.js` : `@wlearn/${pkg}`)
  }
  const preprocessModule = await import(
    portsDir
      ? `${fixturesDir}/../js/preprocess/src/index.js`
      : '@wlearn/preprocess'
  )
  const preprocess = preprocessModule.default || preprocessModule
  await preprocess.registerPreprocess()
}

async function main() {
  const actual = readdirSync(fixturesDir)
    .filter(name => name.endsWith('.wlrn'))
    .map(name => name.slice(0, -5))
    .sort()
  assert.deepEqual(actual, [...expected].sort(), 'golden .wlrn corpus differs from expected set')

  for (const name of expected) {
    assert(existsSync(join(fixturesDir, `${name}.json`)), `missing ${name}.json`)
  }

  await importModels()

  let passed = 0
  for (const name of expected) {
    const bytes = readFileSync(join(fixturesDir, `${name}.wlrn`))
    const sidecar = JSON.parse(readFileSync(join(fixturesDir, `${name}.json`), 'utf8'))
    const { manifest, toc } = validateBundle(bytes, { allowLegacyManifest: false })

    assert.equal(manifest.typeId, sidecar.typeId, `${name}: typeId`)
    assert.deepEqual(manifest.requires || [], sidecar.requires || [], `${name}: requires`)
    if (manifest.typeId !== 'wlearn.pipeline@1') {
      assert.deepEqual(manifest.params, sidecar.params, `${name}: params`)
    }
    assert.equal(toc.length, sidecar.toc.length, `${name}: TOC length`)
    for (let i = 0; i < toc.length; i++) {
      assert.equal(toc[i].id, sidecar.toc[i].id, `${name}: artifact id`)
      assert.equal(toc[i].length, sidecar.toc[i].length, `${name}: artifact length`)
      assert.equal(toc[i].sha256, sidecar.toc[i].sha256, `${name}: artifact hash`)
    }

    const model = await load(bytes)
    try {
      const result = sidecar.operation === 'transform'
        ? await model.transform(sidecar.X)
        : await model.predict(sidecar.X)
      const values = sidecar.operation === 'transform' ? result.data : result
      assertPredictionParity(values, sidecar.predictions, 1e-5, `${name}: ${sidecar.operation || 'predict'}`)
      if (sidecar.operation === 'transform') {
        assert.deepEqual(
          [result.rows, result.cols],
          sidecar.outputShape,
          `${name}: output shape`
        )
      }
      const expectedClasses = sidecar.classes || sidecar.metadata?.classes
      if (Array.isArray(expectedClasses) && expectedClasses.length > 0) {
        const classes = typeof model.classes === 'function' ? model.classes() : model.classes
        assert.deepEqual(Array.from(classes || []), expectedClasses, `${name}: class order`)
      }
    } finally {
      if (typeof model.dispose === 'function') model.dispose()
    }
    console.log(`PASS ${name}`)
    passed++
  }

  console.log(`\nJS golden verification: ${passed}/${expected.length} bundles passed`)
}

main().catch(error => {
  console.error(error.stack || error)
  process.exit(1)
})
