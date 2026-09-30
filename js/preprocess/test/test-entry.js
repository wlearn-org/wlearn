'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const test = require('node:test')

const source = fs.readFileSync(path.join(__dirname, '../src/index.js'), 'utf8')
function loadEntry(available) {
  const native = { hasNativePreparedTransforms: () => available }
  const wasm = { Preprocessor: class {} }
  const result = { module: { exports: {} }, require(id) {
    if (id === 'tranfi') return native
    if (id === './wasm.js') return wasm
    if (id === './factory.js') return { createPreprocessAPI(loader, name) {
      assert.equal(loader(), native)
      return { backend: name }
    } }
    throw new Error(`unexpected import ${id}`)
  } }
  vm.runInNewContext(source, result)
  return { api: result.module.exports, wasm }
}

test('Node entry selects native when prepared ABI is available', () => {
  assert.equal(loadEntry(true).api.backend, 'native')
})

test('Node fallback reuses the existing WASM export identity', () => {
  const { api, wasm } = loadEntry(false)
  assert.equal(api, wasm)
})
