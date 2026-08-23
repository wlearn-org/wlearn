#!/usr/bin/env node

// Verify Python-produced bundles load in JS and produce identical predictions.
// This completes the JS -> Py -> JS round-trip.

import assert from 'node:assert/strict'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeBundle, validateBundle, load } from '@wlearn/core'
import { assertPredictionParity } from './verify-utils.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PY_DIR = join(__dirname, 'py-produced')
const PORTS_DIR = process.env.WLEARN_PORTS_DIR
const REQUIRE_ALL_BACKENDS = process.env.WLEARN_INTEROP_REQUIRE_ALL === '1'

async function tryLoadModels() {
  if (PORTS_DIR) {
    await import(`${PORTS_DIR}/liblinear-wasm/src/index.js`)
    await import(`${PORTS_DIR}/libsvm-wasm/src/index.js`)
    await import(`${PORTS_DIR}/xgboost-wasm/src/index.js`)
    await import(`${PORTS_DIR}/nanoflann-wasm/src/index.js`)
    await import(`${PORTS_DIR}/ebm-wasm/src/index.js`)
    await import(`${PORTS_DIR}/lightgbm-wasm/src/index.js`)
    await import(`${PORTS_DIR}/stochtree-wasm/src/index.js`)
    await import(`${PORTS_DIR}/xlearn-wasm/src/index.js`)
  } else {
    await import('@wlearn/liblinear')
    await import('@wlearn/libsvm')
    await import('@wlearn/xgboost')
    await import('@wlearn/nanoflann')
    await import('@wlearn/ebm')
    await import('@wlearn/lightgbm')
    await import('@wlearn/stochtree')
    await import('@wlearn/xlearn')
  }
  const preprocessModule = await import(
    PORTS_DIR
      ? `${__dirname}/../packages/preprocess/src/index.js`
      : '@wlearn/preprocess'
  )
  const preprocess = preprocessModule.default || preprocessModule
  await preprocess.registerPreprocess()
}

async function main() {
  if (!existsSync(PY_DIR)) {
    throw new Error('Missing py-produced/ directory. Run the Python round-trip test first.')
  }

  const indexPath = join(PY_DIR, 'index.json')
  if (!existsSync(indexPath)) {
    throw new Error('Missing py-produced/index.json. Run the Python round-trip test first.')
  }
  const index = JSON.parse(readFileSync(indexPath, 'utf8'))
  assert(Array.isArray(index.expected), 'py-produced index missing expected[]')
  assert(Array.isArray(index.produced), 'py-produced index missing produced[]')
  assert(Array.isArray(index.skipped), 'py-produced index missing skipped[]')
  if (REQUIRE_ALL_BACKENDS) {
    assert.equal(index.mode, 'full', 'full interop requires a full Python output index')
    assert.deepEqual(index.skipped, [], 'full interop must not skip model backends')
  }
  assert(index.expected.length > 0, 'Python round-trip produced no testable model bundles')
  assert.deepEqual(index.produced, index.expected, 'Python did not produce every expected bundle')

  const files = readdirSync(PY_DIR).filter(f => f.endsWith('.wlrn')).sort()
  assert.deepEqual(files, index.expected.map(name => `${name}.wlrn`).sort(), 'py-produced files differ from index')

  await tryLoadModels()

  let passed = 0, failed = 0

  for (const file of files) {
    const name = file.replace('.wlrn', '')
    try {
      const pyBundle = readFileSync(join(PY_DIR, file))
      const jsBundle = readFileSync(join(__dirname, file))

      // 1. Validate Python bundle format
      validateBundle(pyBundle, { allowLegacyManifest: false })

      // 2. Compare manifests
      const { manifest: pyM, toc: pyToc } = decodeBundle(pyBundle)
      const { manifest: jsM, toc: jsToc } = decodeBundle(jsBundle)

      assert.deepEqual(pyM, jsM, 'decoded manifests differ')

      // 3. Compare complete TOC semantics and blob hashes. Python loaders must
      // preserve upstream bytes when re-saving an unchanged loaded model.
      assert.deepEqual(pyToc, jsToc, 'decoded TOCs differ')

      // 4. Load and compare predictions
      const sidecar = JSON.parse(readFileSync(join(__dirname, `${name}.json`), 'utf8'))
      const model = await load(pyBundle)
      try {
        const result = sidecar.operation === 'transform'
          ? await model.transform(sidecar.X)
          : await model.predict(sidecar.X)
        const values = sidecar.operation === 'transform' ? result.data : result
        assertPredictionParity(values, sidecar.predictions, 1e-5)
        if (sidecar.operation === 'transform') {
          assert.deepEqual([result.rows, result.cols], sidecar.outputShape)
        }
        const expectedClasses = sidecar.classes || sidecar.metadata?.classes
        if (Array.isArray(expectedClasses) && expectedClasses.length > 0) {
          const classes = typeof model.classes === 'function' ? model.classes() : model.classes
          assert.deepEqual(Array.from(classes || []), expectedClasses, 'class order differs')
        }
      } finally {
        if (typeof model.dispose === 'function') model.dispose()
      }

      console.log(`  PASS  ${name}`)
      passed++
    } catch (err) {
      console.log(`  FAIL  ${name}: ${err.message}`)
      failed++
    }
  }

  console.log(`\n${passed} passed, ${failed} failed out of ${files.length} py-produced bundles.`)
  if (failed > 0) process.exit(1)
}

main().catch(err => { console.error(err); process.exit(1) })
