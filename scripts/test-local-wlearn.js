const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')

function sandbox(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wlearn-link-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const repo = path.join(root, 'wlearn')
  fs.mkdirSync(path.join(repo, 'scripts'), { recursive: true })
  fs.copyFileSync(path.join(__dirname, 'link-local-wlearn.js'), path.join(repo, 'scripts/link-local-wlearn.js'))
  const pkg = (dir, name, version, dependencies = {}) => {
    fs.mkdirSync(path.join(root, dir), { recursive: true })
    fs.writeFileSync(path.join(root, dir, 'package.json'), JSON.stringify({ name, version, dependencies }))
  }
  pkg('basis/js', '@wlearn/basis', '0.1.0', { '@wlearn/core': '^0.3.0' })
  pkg('wlearn/js/core', '@wlearn/core', '0.3.0')
  fs.writeFileSync(path.join(repo, 'package.json'), JSON.stringify({ private: true, workspaces: ['js/*'] }))
  const link = path.join(repo, 'node_modules/@wlearn/basis')
  const run = (...args) => spawnSync(process.execPath, [path.join(repo, 'scripts/link-local-wlearn.js'), ...args], {
    encoding: 'utf8', env: { ...process.env, WLEARN_REPOS_ROOT: root }
  })
  return { root, repo, pkg, link, run }
}

test('linker creates absent links, is idempotent, and unlinks owned entries', t => {
  const s = sandbox(t)
  assert.equal(s.run().status, 0)
  assert.equal(fs.realpathSync(s.link), path.join(s.root, 'basis/js'))
  assert.match(s.run().stdout, /already\s+@wlearn\/basis/)
  assert.equal(s.run('--unlink').status, 0)
  assert.equal(fs.existsSync(s.link), false)
})

test('linker repairs dangling links and preserves unowned directories unless forced', t => {
  const s = sandbox(t)
  fs.mkdirSync(path.dirname(s.link), { recursive: true })
  fs.symlinkSync(path.join(s.root, 'missing'), s.link)
  assert.equal(s.run().status, 0)
  assert.equal(fs.realpathSync(s.link), path.join(s.root, 'basis/js'))
  fs.unlinkSync(s.link)
  fs.mkdirSync(s.link)
  fs.writeFileSync(path.join(s.link, 'owned'), 'keep')
  assert.match(s.run().stdout, /existing non-symlink/)
  assert.equal(fs.readFileSync(path.join(s.link, 'owned'), 'utf8'), 'keep')
  assert.equal(s.run('--force').status, 0)
  assert.equal(fs.realpathSync(s.link), path.join(s.root, 'basis/js'))
})

test('dependency check compares declared ranges without creating package links', t => {
  const s = sandbox(t)
  assert.equal(s.run('--check').status, 0)
  assert.equal(fs.existsSync(path.join(s.repo, 'node_modules')), false)
  s.pkg('wlearn/js/sdk', '@wlearn/sdk', '0.3.0', { '@wlearn/core': '0.2.0' })
  const result = s.run('--check')
  assert.equal(result.status, 1)
  assert.match(result.stdout + result.stderr, /@wlearn\/sdk.*@wlearn\/core.*0\.2\.0.*0\.3\.0/)
})

test('dependency check handles caret zero versions and unions, and rejects unsupported syntax', t => {
  const s = sandbox(t)
  for (const [range, ok] of [['^0.2.0 || ^0.3.0', true], ['^0.0.3', false], ['^0.2.0', false], ['0.3.0', true], ['garbage', false]]) {
    s.pkg('basis/js', '@wlearn/basis', '0.1.0', { '@wlearn/core': range })
    assert.equal(s.run('--check').status, ok ? 0 : 1, range)
  }
})

test('dependency check reports installed split core identities without importing them', t => {
  const s = sandbox(t)
  const core = path.join(s.repo, 'js/core/index.js')
  fs.writeFileSync(core, 'throw new Error("must not import")')
  s.pkg('basis/js/node_modules/@wlearn/core', '@wlearn/core', '0.3.0')
  fs.writeFileSync(path.join(s.root, 'basis/js/node_modules/@wlearn/core/index.js'), 'throw new Error("must not import")')
  const result = s.run('--check')
  assert.equal(result.status, 1)
  assert.match(result.stderr, /@wlearn\/basis.*different installed core/)
})
