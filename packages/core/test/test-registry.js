const { describe, it, beforeEach } = require('node:test')
const assert = require('node:assert/strict')
const { register, load, loadSync, getRegistry } = require('../src/registry.js')
const { encodeBundle } = require('../src/bundle.js')
const { RegistryError } = require('../src/errors.js')
const { BundleError } = require('../src/errors.js')

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

  it('validates registration metadata', () => {
    assert.throws(
      () => register('wlearn.test.bad-options@1', () => {}, { sync: 'yes' }),
      RegistryError
    )
    assert.throws(
      () => register('wlearn.test.unknown-option@1', () => {}, { other: true }),
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

  it('passes one immutable context only to opt-in loaders', async () => {
    const runtimeOptions = { maxPlanBytes: 123 }
    let receivedContext
    register('wlearn.test.context@1', function(manifest, toc, blobs, context) {
      assert.equal(arguments.length, 4)
      receivedContext = context
      return context.loaderOptions['wlearn.test.context@1']
    }, { acceptsContext: true })
    const result = await load(makeBundle('wlearn.test.context@1'), {
      loaderOptions: { 'wlearn.test.context@1': runtimeOptions }
    })
    assert.strictEqual(result, runtimeOptions)
    assert(Object.isFrozen(receivedContext))
    assert(Object.isFrozen(receivedContext.loaderOptions))

    register('wlearn.test.no-context@1', function() {
      assert.equal(arguments.length, 3)
      return 'ok'
    })
    assert.equal(await load(makeBundle('wlearn.test.no-context@1'), {
      loaderOptions: { ignored: true }
    }), 'ok')
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

  it('verifies artifact hashes before invoking a loader', async () => {
    let called = false
    register('wlearn.test.corrupt-loader@1', () => { called = true })
    const bytes = makeBundle('wlearn.test.corrupt-loader@1')
    bytes[bytes.length - 1] ^= 0xff
    await assert.rejects(() => load(bytes), BundleError)
    assert.equal(called, false)
  })

  it('fails before dispatch when a declared nested loader is missing', async () => {
    let called = false
    register('wlearn.test.requires@1', () => { called = true })
    const bytes = encodeBundle({
      typeId: 'wlearn.test.requires@1',
      requires: ['wlearn.not-registered@1']
    }, [])
    await assert.rejects(() => load(bytes), /Missing required loader/)
    assert.equal(called, false)
  })
})

describe('loadSync', () => {
  it('works with sync loader', () => {
    const mockEstimator = { type: 'sync' }
    register('wlearn.test.sync-only@1', () => mockEstimator, { sync: true })
    const bytes = makeBundle('wlearn.test.sync-only@1')
    assert.strictEqual(loadSync(bytes), mockEstimator)
  })

  it('rejects an undeclared async loader before invocation', () => {
    let calls = 0
    register('wlearn.test.async-only@1', async () => {
      calls++
      return { type: 'async' }
    })
    const bytes = makeBundle('wlearn.test.async-only@1')
    assert.throws(() => loadSync(bytes), RegistryError)
    assert.equal(calls, 0)
  })

  it('rejects a sync registration contract violation', () => {
    register('wlearn.test.bad-sync-contract@1', async () => ({ type: 'async' }), {
      sync: true
    })
    const bytes = makeBundle('wlearn.test.bad-sync-contract@1')
    assert.throws(() => loadSync(bytes), /contract violation/)
  })

  it('throws for missing loader', () => {
    const bytes = makeBundle('wlearn.no-such@1')
    assert.throws(() => loadSync(bytes), RegistryError)
  })

  it('verifies artifact hashes before invoking a loader', () => {
    let called = false
    register('wlearn.test.corrupt-sync-loader@1', () => { called = true }, {
      sync: true
    })
    const bytes = makeBundle('wlearn.test.corrupt-sync-loader@1')
    bytes[bytes.length - 1] ^= 0xff
    assert.throws(() => loadSync(bytes), BundleError)
    assert.equal(called, false)
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
