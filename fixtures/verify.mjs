#!/usr/bin/env node

// Verify golden fixtures: decode, validate hashes, check manifest fields,
// and (if model packages are available) load via registry and run predictions.
//
// Runs in CI without model packages (bundle format checks only).
// Runs locally with model packages for full prediction verification.

import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeBundle, validateBundle, load } from '@wlearn/core'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PORTS_DIR = process.env.WLEARN_PORTS_DIR

// --- Import model packages (optional, for prediction checks) ---

let modelsAvailable = false

async function tryLoadModels() {
  try {
    if (PORTS_DIR) {
      await import(`${PORTS_DIR}/liblinear-wasm/src/index.js`)
      await import(`${PORTS_DIR}/libsvm-wasm/src/index.js`)
      await import(`${PORTS_DIR}/xgboost-wasm/src/index.js`)
      await import(`${PORTS_DIR}/nanoflann-wasm/src/index.js`)
      await import(`${PORTS_DIR}/ebm-wasm/src/index.js`)
    } else {
      await import('@wlearn/liblinear')
      await import('@wlearn/libsvm')
      await import('@wlearn/xgboost')
      await import('@wlearn/nanoflann')
      await import('@wlearn/ebm')
    }
    modelsAvailable = true
  } catch {
    console.log('Model packages not available -- skipping prediction checks.\n')
  }
}

// --- Helpers ---

function assertEq(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

function assertClose(actual, expected, tol, msg) {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${msg}: expected ~${expected}, got ${actual} (tol=${tol})`)
  }
}

// --- Main ---

async function main() {
  await tryLoadModels()

  const files = readdirSync(__dirname).filter(f => f.endsWith('.wlrn'))
  if (files.length === 0) {
    console.error('No .wlrn fixtures found. Run generate.mjs first.')
    process.exit(1)
  }

  let passed = 0
  let failed = 0

  for (const file of files) {
    const name = file.replace('.wlrn', '')
    const wlrnPath = join(__dirname, file)
    const jsonPath = join(__dirname, `${name}.json`)

    try {
      const bundle = readFileSync(wlrnPath)
      const sidecar = JSON.parse(readFileSync(jsonPath, 'utf8'))

      // 1. Decode
      const { manifest, toc, blobs } = decodeBundle(bundle)

      // 2. Validate hashes
      validateBundle(bundle)

      // 3. Check manifest fields match sidecar
      assertEq(manifest.typeId, sidecar.typeId, `${name}: typeId mismatch`)

      // 4. Check TOC entries match sidecar
      assertEq(toc.length, sidecar.toc.length, `${name}: toc length mismatch`)
      for (let i = 0; i < toc.length; i++) {
        assertEq(toc[i].id, sidecar.toc[i].id, `${name}: toc[${i}].id mismatch`)
        assertEq(toc[i].length, sidecar.toc[i].length, `${name}: toc[${i}].length mismatch`)
        assertEq(toc[i].sha256, sidecar.toc[i].sha256, `${name}: toc[${i}].sha256 mismatch`)
      }

      // 5. Load via registry and verify predictions (if models available)
      if (modelsAvailable) {
        const model = await load(bundle)
        const X = sidecar.X
        const preds = model.predict(X)
        const expected = sidecar.predictions

        assertEq(preds.length, expected.length, `${name}: prediction count mismatch`)
        for (let i = 0; i < preds.length; i++) {
          assertClose(preds[i], expected[i], 1e-5, `${name}: pred[${i}]`)
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

  console.log(`\n${passed} passed, ${failed} failed out of ${files.length} fixtures.`)
  if (failed > 0) process.exit(1)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
