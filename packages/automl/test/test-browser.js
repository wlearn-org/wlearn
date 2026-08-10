#!/usr/bin/env node
// Browser smoke test for IIFE + ESM bundles
// Generic: auto-discovers package name and exports from package.json + src/index.js

const path = require('path')
const http = require('http')
const fs = require('fs')
const { encodeBundle } = require('@wlearn/core')

const ROOT = path.resolve(__dirname, '..')
const pkg = require(path.join(ROOT, 'package.json'))
const NAME = pkg.name.split('/').pop()
const EXPORTS = Object.keys(require(path.join(ROOT, 'src', 'index.js')))
const TMP_DIR = fs.mkdtempSync(path.join(ROOT, '.browser-test-'))
let chromium
const PROBE_MODEL_BUNDLE = Buffer.from(encodeBundle({
  typeId: 'wlearn.ensemble.voting.classifier@1',
  params: {
    task: 'classification',
    voting: 'soft',
    weights: [],
    estimatorNames: [],
    classes: [0, 1],
  },
}, [])).toString('base64')

function failMissingPlaywright(e) {
  fs.rmSync(TMP_DIR, { recursive: true, force: true })
  console.error(e.message)
  console.error(`\nBrowser tests use the core repo Playwright install and Chromium cache.\nRun from /home/anton/projects/wlearn/wlearn with PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH set, or install Chromium once in the core repo.`)
  process.exit(1)
}

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
      .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))
      .map(name => path.join(cacheRoot, name, 'chrome-linux64', 'chrome'))
      .filter(executableExists)
    if (candidates.length) return candidates[0]
  } catch (_) {}

  return undefined
}

try {
  chromium = require('playwright').chromium
} catch (e) {
  failMissingPlaywright(e)
}

const bundles = [
  { name: 'IIFE', file: `dist/${NAME}.js`,  type: 'iife', global: NAME },
  { name: 'ESM',  file: `dist/${NAME}.mjs`, type: 'esm' },
]

function makeIifeHtml(jsPath, globalName, exportKeys) {
  return `<!DOCTYPE html><html><body>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    var lib = ${globalName}
    var expected = ${JSON.stringify(exportKeys)}
    var missing = expected.filter(function(k) { return !(k in lib) })
    if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
    var types = {}
    expected.forEach(function(k) { types[k] = typeof lib[k] })
    var execution = await runPreprocessAutoFit(lib)
    return { ok: true, exports: expected.length, types: types, execution: execution }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
${browserExecutionSource(PROBE_MODEL_BUNDLE)}
window.__testResult = runTest()
</script></body></html>`
}

function makeEsmHtml(jsPath, exportKeys) {
  const imports = exportKeys.join(', ')
  return `<!DOCTYPE html><html><body>
<script type="module">
import { ${imports} } from '${jsPath}'
async function runTest() {
  try {
    var types = {}
    var exports = [${exportKeys.map(k => `['${k}', ${k}]`).join(', ')}]
    exports.forEach(function(e) { types[e[0]] = typeof e[1] })
    var execution = await runPreprocessAutoFit({ autoFit: autoFit })
    return { ok: true, exports: ${exportKeys.length}, types: types, execution: execution }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
${browserExecutionSource(PROBE_MODEL_BUNDLE)}
window.__testResult = runTest()
</script></body></html>`
}

function browserExecutionSource(probeModelBundle) {
  return `
class BrowserProbeModel {
  static get classId() { return 'wlearn.test.browser-probe@1' }
  static defaultSearchSpace() { return {} }
  static async create(params) { return new BrowserProbeModel(params) }
  constructor(params) { this.params = Object.assign({}, params); this.fitted = false }
  fit(X, y) {
    var counts = new Map()
    for (var i = 0; i < y.length; i++) counts.set(y[i], (counts.get(y[i]) || 0) + 1)
    this.label = 0
    var best = -1
    counts.forEach((count, label) => {
      if (count > best) { best = count; this.label = label }
    })
    this.fitted = true
    return this
  }
  predict(X) { return new Int32Array(X.rows).fill(this.label) }
  predictProba(X) {
    var out = new Float64Array(X.rows * 2)
    for (var i = 0; i < X.rows; i++) out[i * 2 + this.label] = 1
    return out
  }
  getParams() { return Object.assign({}, this.params) }
  save() {
    var binary = atob('${probeModelBundle}')
    var bytes = new Uint8Array(binary.length)
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
    return bytes
  }
  setParams(params) { Object.assign(this.params, params); return this }
  dispose() { this.fitted = false }
  get isFitted() { return this.fitted }
  get capabilities() { return { classifier: true, regressor: false } }
}
async function runPreprocessAutoFit(lib) {
  var X = {
    data: new Float64Array([0.1, 0.2, 0.3, 0.4, 10.1, 20.2, 30.3, 40.4]),
    rows: 8,
    cols: 1
  }
  var y = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1])
  var loadOptions = {
    loaderOptions: {
      'wlearn.preprocess.tabular@1': { limits: { maxPlanBytes: 1048576 } }
    }
  }
  var result = await lib.autoFit(
    [{ name: 'browser-probe', cls: BrowserProbeModel }],
    X,
    y,
    {
      nIter: 1,
      cv: 2,
      task: 'classification',
      preprocess: { encode: false, impute: false, scale: 'standard' },
      ensemble: false,
      refit: true
    }
  )
  var loaded = null
  var nested = null
  var loadedNested = null
  try {
    var predictions = result.model.predict(X)
    if (!(predictions instanceof Int32Array) || predictions.length !== X.rows) {
      throw new Error('browser AutoML preprocessing returned invalid predictions')
    }
    if (!result.bestCandidate || !result.bestCandidate.preprocess) {
      throw new Error('browser AutoML result omitted preprocessing candidate identity')
    }
    if (!/^wlc1_[0-9a-f]{64}$/.test(result.leaderboard[0].candidateId)) {
      throw new Error('browser AutoML result returned a noncanonical candidate ID')
    }
    nested = await lib.autoFit(
      [
        { name: 'browser-one', classId: 'wlearn.test.browser-one@1', cls: BrowserProbeModel },
        { name: 'browser-two', classId: 'wlearn.test.browser-two@1', cls: BrowserProbeModel }
      ],
      X,
      y,
      {
        nIter: 1,
        cv: 2,
        task: 'classification',
        preprocess: { encode: false, impute: false, scale: 'standard' },
        ensemble: true,
        ensembleSize: 2
      }
    )
    var bytes = result.model.save()
    loaded = await result.model.constructor.load(bytes, loadOptions)
    var loadedPredictions = loaded.predict(X)
    if (loadedPredictions.length !== X.rows ||
        loaded.provenance.candidateId !== result.leaderboard[0].candidateId) {
      throw new Error('browser AutoML Pipeline round-trip lost predictions or provenance')
    }
    var nestedBytes = nested.model.save()
    loadedNested = await nested.model.constructor.load(nestedBytes, loadOptions)
    var nestedPredictions = loadedNested.predict(X)
    if (!(nestedPredictions instanceof Float64Array) || nestedPredictions.length !== X.rows) {
      throw new Error('browser nested AutoML ensemble round-trip returned invalid predictions')
    }
    return {
      rows: loadedPredictions.length,
      nestedRows: nestedPredictions.length,
      templateId: result.bestCandidate.preprocess.templateId
    }
  } finally {
    if (loadedNested) loadedNested.dispose()
    if (nested && nested.model) nested.model.dispose()
    if (loaded) loaded.dispose()
    if (result.model) result.model.dispose()
  }
}`
}

async function main() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || '/').split('?')[0])
    const fp = path.resolve(ROOT, url.replace(/^\/+/, ''))
    if (!fp.startsWith(ROOT + path.sep)) {
      res.writeHead(403)
      res.end('Forbidden')
      return
    }
    if (!fs.existsSync(fp)) { res.writeHead(404); res.end('Not found: ' + req.url); return }
    const ext = path.extname(fp)
    const ct = ext === '.html' ? 'text/html' : ext === '.mjs' ? 'text/javascript' : 'application/javascript'
    res.writeHead(200, { 'Content-Type': ct })
    res.end(fs.readFileSync(fp))
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  const port = server.address().port
  const base = `http://127.0.0.1:${port}`

  let browser
  try {
    const executablePath = chromiumExecutablePath()
    browser = await chromium.launch({ headless: true, executablePath })
  } catch (e) {
    server.close()
    failMissingPlaywright(e)
  }
  let passed = 0, failed = 0

  for (const b of bundles) {
    const htmlName = `_test_${b.name}.html`
    const htmlPath = path.join(TMP_DIR, htmlName)
    const jsUrl = '/' + b.file
    const htmlUrl = '/' + path.relative(ROOT, htmlPath).replace(/\\/g, '/')

    if (b.type === 'iife') {
      fs.writeFileSync(htmlPath, makeIifeHtml(jsUrl, b.global, EXPORTS))
    } else {
      fs.writeFileSync(htmlPath, makeEsmHtml(jsUrl, EXPORTS))
    }

    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(e.message))

    try {
      await page.goto(`${base}${htmlUrl}`, { timeout: 30000 })
      await page.waitForFunction(() => window.__testResult, { timeout: 30000 })
      const result = await page.evaluate(() => window.__testResult)

      if (result && result.ok) {
        console.log(
          `  PASS: ${b.name} -- ${result.exports} exports, ` +
          `${result.execution.rows} direct + ${result.execution.nestedRows} nested rows`
        )
        passed++
      } else {
        console.log(`  FAIL: ${b.name} -- ${result ? result.error : 'no result'}`)
        if (result && result.stack) console.log(`        ${result.stack.split('\n')[1]}`)
        failed++
      }
    } catch (e) {
      console.log(`  FAIL: ${b.name} -- ${e.message}`)
      if (errors.length) console.log(`        page errors: ${errors.join('; ')}`)
      failed++
    }

    await page.close()
  }

  await browser.close()
  server.close()
  fs.rmSync(TMP_DIR, { recursive: true, force: true })

  console.log(`\n=== ${passed} passed, ${failed} failed ===`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(e => {
  console.error(e)
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }) } catch (_) {}
  process.exit(1)
})
