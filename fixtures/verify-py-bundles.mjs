#!/usr/bin/env node

// Verify Python-produced bundles load in JS and produce identical predictions.
// This completes the JS -> Py -> JS round-trip.

import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeBundle, validateBundle, load } from '@wlearn/core'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PY_DIR = join(__dirname, 'py-produced')
const PORTS_DIR = process.env.WLEARN_PORTS_DIR

async function tryLoadModels() {
  try {
    if (PORTS_DIR) {
      await import(`${PORTS_DIR}/liblinear-wasm/src/index.js`)
      await import(`${PORTS_DIR}/libsvm-wasm/src/index.js`)
      await import(`${PORTS_DIR}/xgboost-wasm/src/index.js`)
      await import(`${PORTS_DIR}/nanoflann-wasm/src/index.js`)
      await import(`${PORTS_DIR}/ebm-wasm/src/index.js`)
      await import(`${PORTS_DIR}/lightgbm-wasm/src/index.js`)
    } else {
      await import('@wlearn/liblinear')
      await import('@wlearn/libsvm')
      await import('@wlearn/xgboost')
      await import('@wlearn/nanoflann')
      await import('@wlearn/ebm')
      await import('@wlearn/lightgbm')
    }
    return true
  } catch {
    console.log('Model packages not available -- cannot verify predictions.\n')
    return false
  }
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) {
    throw new Error(`${msg}: expected ~${b}, got ${a} (tol=${tol})`)
  }
}

async function main() {
  if (!existsSync(PY_DIR)) {
    console.log('No py-produced/ directory. Run the Python round-trip test first.')
    process.exit(0)
  }

  const modelsAvailable = await tryLoadModels()
  const files = readdirSync(PY_DIR).filter(f => f.endsWith('.wlrn'))
  if (files.length === 0) {
    console.log('No .wlrn files in py-produced/.')
    process.exit(0)
  }

  let passed = 0, failed = 0

  for (const file of files) {
    const name = file.replace('.wlrn', '')
    try {
      const pyBundle = readFileSync(join(PY_DIR, file))
      const jsBundle = readFileSync(join(__dirname, file))

      // 1. Validate Python bundle format
      validateBundle(pyBundle)

      // 2. Compare manifests
      const { manifest: pyM, toc: pyToc } = decodeBundle(pyBundle)
      const { manifest: jsM, toc: jsToc } = decodeBundle(jsBundle)

      if (pyM.typeId !== jsM.typeId) throw new Error(`typeId: ${pyM.typeId} !== ${jsM.typeId}`)
      if (JSON.stringify(pyM.params) !== JSON.stringify(jsM.params)) {
        throw new Error(`params differ`)
      }

      // 3. Compare blob hashes (should be identical for most models)
      // LightGBM text format is not byte-stable across C API (WASM) vs Python
      // save_model() -- skip hash check, rely on prediction verification instead.
      const skipBlobHash = name.startsWith('lightgbm-')
      if (!skipBlobHash) {
        for (let i = 0; i < jsToc.length; i++) {
          if (pyToc[i].sha256 !== jsToc[i].sha256) {
            throw new Error(`blob "${jsToc[i].id}" sha256 differs`)
          }
        }
      }

      // 4. Load and compare predictions
      if (modelsAvailable) {
        const sidecar = JSON.parse(readFileSync(join(__dirname, `${name}.json`), 'utf8'))
        const model = await load(pyBundle)
        const preds = model.predict(sidecar.X)
        for (let i = 0; i < preds.length; i++) {
          assertClose(preds[i], sidecar.predictions[i], 1e-5, `pred[${i}]`)
        }
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
