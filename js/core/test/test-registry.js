const { describe, it, beforeEach } = require('node:test')
const assert = require('node:assert/strict')
const { register, load, loadSync, getRegistry } = require('../src/registry.js')
const { encodeBundle } = require('../src/bundle.js')
const { RegistryError } = require('../src/errors.js')
const { BundleError } = require('../src/errors.js')

it('independently evaluated core copies share loaders and preserve nested context', async () => {
  const fs = require('node:fs')
  const { createRequire } = require('node:module')
  const filename = require.resolve('../src/registry.js')
  const source = fs.readFileSync(filename, 'utf8')
  const localRequire = createRequire(filename)
  function copy(version) {
    const module = { exports: {} }
    const require = id => id === '../package.json' && version
      ? { version } : localRequire(id)
    new Function('require', 'module', source)(require, module)
    return module.exports
  }
  const first = copy()
  const second = copy()
  const contextValue = { threshold: 3 }
  first.register('wlearn.test.independent-child@1', (_m, _t, _b, ctx) => ctx,
    { acceptsContext: true })
  second.register('wlearn.test.independent-parent@1', (_m, _t, _b, ctx) =>
    first.load(makeBundle('wlearn.test.independent-child@1'), ctx), { acceptsContext: true })
  const result = await first.load(makeBundle('wlearn.test.independent-parent@1'), {
    loaderOptions: { child: contextValue }
  })
  assert.deepEqual(result.loaderOptions.child, contextValue)
  assert(Object.isFrozen(result.loaderOptions.child))
  assert.throws(() => copy('999.0.0'), /core.*version|version.*core/i)
})

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
    const cancelFlag = new Int32Array(new SharedArrayBuffer(4))
    const runtimeOptions = { limits: { maxPlanBytes: 123 }, cancelFlag }
    let receivedContext
    register('wlearn.test.context@1', function(manifest, toc, blobs, context) {
      assert.equal(arguments.length, 4)
      receivedContext = context
      return context.loaderOptions['wlearn.test.context@1']
    }, { acceptsContext: true })
    const result = await load(makeBundle('wlearn.test.context@1'), {
      loaderOptions: { 'wlearn.test.context@1': runtimeOptions }
    })
    assert.deepEqual(result.limits, runtimeOptions.limits)
    assert.notStrictEqual(result, runtimeOptions)
    assert.notStrictEqual(result.limits, runtimeOptions.limits)
    assert.strictEqual(result.cancelFlag, cancelFlag)
    assert(Object.isFrozen(receivedContext))
    assert(Object.isFrozen(receivedContext.loaderOptions))
    assert(Object.isFrozen(result))
    assert(Object.isFrozen(result.limits))

    register('wlearn.test.no-context@1', function() {
      assert.equal(arguments.length, 3)
      return 'ok'
    })
    assert.equal(await load(makeBundle('wlearn.test.no-context@1'), {
      loaderOptions: { ignored: true }
    }), 'ok')
  })

  it('snapshots nested loader options before an async loader runs', async () => {
    let release
    const gate = new Promise(resolve => { release = resolve })
    register('wlearn.test.context-race@1', async (manifest, toc, blobs, context) => {
      await gate
      return context.loaderOptions['wlearn.test.context-race@1'].limits.maxPlanBytes
    }, { acceptsContext: true })
    const runtimeOptions = { limits: { maxPlanBytes: 123 } }
    const pending = load(makeBundle('wlearn.test.context-race@1'), {
      loaderOptions: { 'wlearn.test.context-race@1': runtimeOptions }
    })
    runtimeOptions.limits.maxPlanBytes = 1
    release()
    assert.equal(await pending, 123)
  })

  it('preserves own __proto__ option keys without mutating prototypes', async () => {
    register('wlearn.test.context-proto@1', (manifest, toc, blobs, context) =>
      context.loaderOptions['wlearn.test.context-proto@1'],
    { acceptsContext: true })
    const runtimeOptions = JSON.parse('{"__proto__":{"safe":true}}')
    const result = await load(makeBundle('wlearn.test.context-proto@1'), {
      loaderOptions: { 'wlearn.test.context-proto@1': runtimeOptions },
    })
    assert(Object.hasOwn(result, '__proto__'))
    assert.deepEqual(result.__proto__, { safe: true })
    assert.equal(Object.getPrototypeOf(result), Object.prototype)
    assert.equal({}.safe, undefined)
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
