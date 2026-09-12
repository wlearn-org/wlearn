#!/usr/bin/env node
'use strict'

const fs = require('node:fs')
const http = require('node:http')
const path = require('node:path')
const esbuild = require('esbuild')

const ROOT = path.resolve(__dirname, '..')
const TMP_DIR = fs.mkdtempSync(path.join(ROOT, '.browser-test-'))
let chromium

function executableExists(file) {
  try {
    fs.accessSync(file, fs.constants.X_OK)
    return true
  } catch (_) {
    return false
  }
}

function chromiumExecutablePath() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH &&
      executableExists(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH)) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  }
  const expected = chromium.executablePath()
  if (executableExists(expected)) return expected
  const cacheRoot = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/ms-playwright'
  try {
    const candidates = fs.readdirSync(cacheRoot)
      .filter(name => /^chromium-\d+$/.test(name))
      .sort((left, right) => Number(right.split('-')[1]) - Number(left.split('-')[1]))
      .map(name => path.join(cacheRoot, name, 'chrome-linux64', 'chrome'))
      .filter(executableExists)
    return candidates[0]
  } catch (_) {
    return undefined
  }
}

function testBody(apiExpression) {
  return `
async function exercise(api) {
  var fixed = await api.Preprocessor.create({columns: {
    x0: {kind: 'numeric', scale: false}, x1: {categories: [5, 0, 2]}
  }})
  fixed.fit([[1, 2], [2, 2]])
  var fixedLoaded = await api.Preprocessor.load(fixed.save())
  var fixedOutput = Array.from(fixedLoaded.transform([[3, 5], [4, 99]]).data)
  if (JSON.stringify(fixedOutput) !== JSON.stringify([3, 0, 0, 1, 4, 0, 0, 0])) {
    throw new Error('fixed dictionary column policy mismatch')
  }
  fixed.dispose()
  fixedLoaded.dispose()
  if (typeof api.StandardScaler !== 'function' ||
      typeof api.MinMaxScaler !== 'function') {
    throw new Error('missing compatibility scaler exports')
  }
  var preprocessor = await api.Preprocessor.create({
    impute: 'median', encode: 'label', scale: 'minmax'
  })
  try {
    var fitted = preprocessor.fitTransform([
      [1, 4.5], [2, 1.5], [1, NaN], [2, 9.5]
    ])
    if (fitted.rows !== 4 || fitted.cols !== 2) throw new Error('bad fitted shape')
    var bundle = preprocessor.save()
    await api.registerPreprocess()
    var restored = await api.Preprocessor.load(bundle)
    try {
      var output = restored.transform([[2, 4.5], [3, 5.5]])
      var values = Array.from(output.data)
      var expected = [1, 0.375, -1, 0.5]
      if (values.length !== expected.length ||
          values.some(function(value, index) { return value !== expected[index] })) {
        throw new Error('bad restored values: ' + values.join(','))
      }
      return { ok: true, bytes: bundle.byteLength, backend: restored.backend }
    } finally {
      restored.dispose()
    }
  } finally {
    preprocessor.dispose()
  }
}
window.__testResult = exercise(${apiExpression}).catch(function(error) {
  return { ok: false, error: error.message, stack: error.stack }
})`
}

function htmlFor(kind) {
  if (kind === 'IIFE') {
    return `<!doctype html><script src="/dist/preprocess.js"></script><script>${testBody('wlearnPreprocess')}</script>`
  }
  return `<!doctype html><script type="module">import * as api from '/dist/preprocess.mjs';${testBody('api')}</script>`
}

async function buildConsumer() {
  const tranfiPath = process.env.TRANFI_WASM_ENTRY ||
    (process.env.TRANFI_WASM_PATH
      ? path.join(process.env.TRANFI_WASM_PATH, 'index.js')
      : null)
  const options = {
    entryPoints: [path.join(__dirname, 'browser-consumer.mjs')],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    minify: true,
    treeShaking: true,
    external: ['node:fs', 'node:crypto'],
    outfile: path.join(TMP_DIR, 'consumer.mjs')
  }
  if (tranfiPath) options.alias = { 'tranfi/wasm': tranfiPath }
  await esbuild.build(options)
}

async function main() {
  try {
    chromium = require('playwright').chromium
  } catch (error) {
    throw new Error(`Playwright is unavailable: ${error.message}`)
  }

  const server = http.createServer((request, response) => {
    const url = decodeURIComponent((request.url || '/').split('?')[0])
    const file = path.resolve(ROOT, url.replace(/^\/+/, ''))
    if (!file.startsWith(ROOT + path.sep) || !fs.existsSync(file)) {
      response.writeHead(404)
      response.end('not found')
      return
    }
    const contentType = file.endsWith('.html')
      ? 'text/html'
      : file.endsWith('.mjs') ? 'text/javascript' : 'application/javascript'
    response.writeHead(200, { 'Content-Type': contentType })
    response.end(fs.readFileSync(file))
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const browser = await chromium.launch({
    headless: true,
    executablePath: chromiumExecutablePath()
  })
  try {
    for (const kind of ['IIFE', 'ESM']) {
      const html = path.join(TMP_DIR, `${kind.toLowerCase()}.html`)
      fs.writeFileSync(html, htmlFor(kind))
      const page = await browser.newPage()
      try {
        await page.goto(
          `http://127.0.0.1:${server.address().port}/${path.relative(ROOT, html)}`,
          { timeout: 30000 }
        )
        await page.waitForFunction(() => window.__testResult, { timeout: 30000 })
        const result = await page.evaluate(() => window.__testResult)
        if (!result || !result.ok) {
          throw new Error(`${kind}: ${result ? result.error : 'no result'}`)
        }
        console.log(`PASS ${kind}: ${result.backend}, ${result.bytes} bundle bytes`)
      } finally {
        await page.close()
      }
    }

    await buildConsumer()
    const consumerHtml = path.join(TMP_DIR, 'consumer.html')
    fs.writeFileSync(consumerHtml, `<!doctype html><script type="module">
      import { runNestedConsumer } from './consumer.mjs'
      window.__testResult = runNestedConsumer()
        .then(function(result) {
          if (result.predictions.join(',') !== '2,2') throw new Error('bad nested predictions')
          if (result.limitCode !== 'ERR_RESOURCE_LIMIT') throw new Error('load context was not forwarded')
          if (!result.scalerIdentity) throw new Error('scaler exports changed identity')
          return { ok: true, bytes: result.bytes }
        })
        .catch(function(error) { return { ok: false, error: error.message, stack: error.stack } })
    </script>`)
    const consumerPage = await browser.newPage()
    try {
      await consumerPage.goto(
        `http://127.0.0.1:${server.address().port}/${path.relative(ROOT, consumerHtml)}`,
        { timeout: 30000 }
      )
      await consumerPage.waitForFunction(() => window.__testResult, { timeout: 30000 })
      const result = await consumerPage.evaluate(() => window.__testResult)
      if (!result || !result.ok) {
        throw new Error(`production consumer: ${result ? result.error : 'no result'}`)
      }
      console.log(`PASS production consumer: nested Pipeline, ${result.bytes} bytes`)
    } finally {
      await consumerPage.close()
    }
  } finally {
    await browser.close()
    server.close()
    fs.rmSync(TMP_DIR, { recursive: true, force: true })
  }
}

main().catch(error => {
  console.error(error)
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }) } catch (_) {}
  process.exit(1)
})
