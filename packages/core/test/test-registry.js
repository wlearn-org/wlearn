const { describe, it, beforeEach } = require('node:test')
const assert = require('node:assert/strict')
const { register, load, loadSync, getRegistry } = require('../src/registry.js')
const { encodeBundle } = require('../src/bundle.js')
const { RegistryError } = require('../src/errors.js')

// Helper: create a minimal bundle
function makeBundle(typeId, data = new Uint8Array([1])) {
  return encodeBundle({ typeId }, [{ id: 'model', data }])
}

describe('register', () => {
  it('validates typeId contains @', () => {
    assert.throws(
      () => register('bad-type-id', () => {}),
      RegistryError
    )
  })

  it('validates loaderFn is a function', () => {
    assert.throws(
      () => register('wlearn.test@1', 'not a function'),
      RegistryError
    )
  })

  it('registers a loader', () => {
    register('wlearn.test.register@1', () => 'ok')
    const reg = getRegistry()
    assert(reg.has('wlearn.test.register@1'))
  })
})

describe('load (async)', () => {
  it('dispatches to sync loader and returns Promise', async () => {
    const mockEstimator = { isFitted: true }
    register('wlearn.test.sync-loader@1', () => mockEstimator)

    const bytes = makeBundle('wlearn.test.sync-loader@1')
    const result = load(bytes)
    assert(result instanceof Promise)
    assert.strictEqual(await result, mockEstimator)
  })

  it('dispatches to async loader', async () => {
    const mockEstimator = { isFitted: true }
    register('wlearn.test.async-loader@1', async () => mockEstimator)

    const bytes = makeBundle('wlearn.test.async-loader@1')
    const result = await load(bytes)
    assert.strictEqual(result, mockEstimator)
  })

  it('throws RegistryError for missing loader', async () => {
    const bytes = makeBundle('wlearn.unknown@1')
    await assert.rejects(() => load(bytes), (err) => {
      assert(err instanceof RegistryError)
      assert(err.message.includes('wlearn.unknown@1'))
      assert(err.message.includes('No loader registered'))
      return true
    })
  })

  it('error includes available loaders', async () => {
    register('wlearn.test.listed@1', () => {})
    const bytes = makeBundle('wlearn.not-here@1')
    await assert.rejects(() => load(bytes), (err) => {
      assert(err.message.includes('wlearn.test.listed@1'))
      return true
    })
  })
})

describe('loadSync', () => {
  it('works with sync loader', () => {
    const mockEstimator = { type: 'sync' }
    register('wlearn.test.sync-only@1', () => mockEstimator)
    const bytes = makeBundle('wlearn.test.sync-only@1')
    assert.strictEqual(loadSync(bytes), mockEstimator)
  })

  it('throws if loader returns Promise', () => {
    register('wlearn.test.async-only@1', async () => ({ type: 'async' }))
    const bytes = makeBundle('wlearn.test.async-only@1')
    assert.throws(() => loadSync(bytes), RegistryError)
  })

  it('throws for missing loader', () => {
    const bytes = makeBundle('wlearn.no-such@1')
    assert.throws(() => loadSync(bytes), RegistryError)
  })
})

describe('getRegistry', () => {
  it('returns a copy', () => {
    register('wlearn.test.copy@1', () => {})
    const reg = getRegistry()
    reg.delete('wlearn.test.copy@1')
    // Internal registry unaffected
    const reg2 = getRegistry()
    assert(reg2.has('wlearn.test.copy@1'))
  })
})
