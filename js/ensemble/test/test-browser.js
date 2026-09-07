#!/usr/bin/env node
// Browser smoke test for IIFE + ESM bundles
// Generic: auto-discovers package name and exports from package.json + src/index.js

const path = require('path')
const http = require('http')
const fs = require('fs')

const ROOT = path.resolve(__dirname, '..')
const pkg = require(path.join(ROOT, 'package.json'))
const NAME = pkg.name.split('/').pop()
const BASIS_ROOT = path.resolve(path.dirname(require.resolve('@wlearn/basis')), '..')
const EXPORTS = Object.keys(require(path.join(ROOT, 'src', 'index.js')))
const TMP_DIR = fs.mkdtempSync(path.join(ROOT, '.browser-test-'))
const CORE_BUNDLE = path.join(TMP_DIR, 'core.js')
require('esbuild').buildSync({
  entryPoints: [require.resolve('@wlearn/core')], bundle: true,
  platform: 'browser', format: 'iife', globalName: 'wlearnCore', outfile: CORE_BUNDLE
})
let chromium

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

async function roundtrip(VotingEnsemble, StackingEnsemble, BasisRegressor, caruanaSelect) {
  let taskError
  try { await BasisRegressor.create({ task: 'classification' }) } catch (error) { taskError = error }
  if (!(taskError instanceof wlearnCore.ValidationError)) throw new Error('Public core error identity differs between bundles')
  wlearnCore.registerMeasure({ id: 'browser-test-loss', taskKinds: ['classification'],
    direction: 'minimize', response: 'proba',
    fn: ({ truth, proba }) => wlearnCore.logLoss(truth, proba, { classes: [0, 1] }) })
  const selection = caruanaSelect([[0.9, 0.1, 0.1, 0.9]], [0, 1], {
    scoring: 'browser-test-loss', maxSize: 1
  })
  if (selection.indices[0] !== 0) throw new Error('Custom measure did not cross the bundle boundary')
  const X = [[-1, 0], [0, 1], [1, 0], [2, 1], [3, 0], [4, 1]]
  const y = [-1, 1, 3, 5, 7, 9]
  for (const Ensemble of [VotingEnsemble, StackingEnsemble]) {
    const params = { method: 'rvfl', nComponents: 4 }
    const model = await Ensemble.create({ task: 'regression',
      estimators: [['basis', BasisRegressor, params]],
      finalEstimator: ['readout', BasisRegressor, params], cv: 2 })
    let restored
    try {
      await model.fit(X, y)
      const before = Array.from(await model.predict(X))
      restored = await Ensemble.load(model.save())
      const after = Array.from(await restored.predict(X))
      if (before.some((value, index) => Math.abs(value - after[index]) > 1e-8)) {
        throw new Error('Composed model predictions changed after loading')
      }
    } finally {
      model.dispose()
      if (restored) restored.dispose()
    }
  }
}

function makeIifeHtml(jsPath, globalName, exportKeys) {
  return `<!DOCTYPE html><html><body>
<script src="/${path.relative(ROOT, CORE_BUNDLE).replace(/\\/g, '/')}"></script>
<script src="/basis.js"></script>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    await (${roundtrip.toString()})(${globalName}.VotingEnsemble, ${globalName}.StackingEnsemble, basis.BasisRegressor, ${globalName}.caruanaSelect)
    var lib = ${globalName}
    var expected = ${JSON.stringify(exportKeys)}
    var missing = expected.filter(function(k) { return !(k in lib) })
    if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
    var types = {}
    expected.forEach(function(k) { types[k] = typeof lib[k] })
    return { ok: true, exports: expected.length, types: types }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

function makeEsmHtml(jsPath, exportKeys) {
  const imports = exportKeys.join(', ')
  return `<!DOCTYPE html><html><body>
<script src="/${path.relative(ROOT, CORE_BUNDLE).replace(/\\/g, '/')}"></script>
<script type="module">
import { ${imports} } from '${jsPath}'
import { BasisRegressor } from '/basis.mjs'
async function runTest() {
  try {
    await (${roundtrip.toString()})(VotingEnsemble, StackingEnsemble, BasisRegressor, caruanaSelect)
    var types = {}
    var exports = [${exportKeys.map(k => `['${k}', ${k}]`).join(', ')}]
    exports.forEach(function(e) { types[e[0]] = typeof e[1] })
    return { ok: true, exports: ${exportKeys.length}, types: types }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

async function main() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || '/').split('?')[0])
    const modelFile = url === '/basis.js' || url === '/basis.mjs'
    const fp = modelFile ? path.join(BASIS_ROOT, 'dist', url.slice(1)) : path.resolve(ROOT, url.replace(/^\/+/, ''))
    if (!modelFile && !fp.startsWith(ROOT + path.sep)) {
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
        console.log(`  PASS: ${b.name} -- ${result.exports} exports`)
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
