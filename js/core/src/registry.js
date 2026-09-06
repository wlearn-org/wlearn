const { RegistryError } = require('./errors.js')
const { validateBundle } = require('./bundle.js')

const registry = new Map()
const LOAD_CONTEXT = Symbol('wlearn.loadContext')

function register(typeId, loaderFn, options = {}) {
  if (typeof typeId !== 'string' || !typeId.includes('@')) {
    throw new RegistryError(`Invalid typeId "${typeId}": must contain "@" (e.g. "wlearn.liblinear.classifier@1")`)
  }
  if (typeof loaderFn !== 'function') {
    throw new RegistryError('loaderFn must be a function')
  }
  if (!isPlainObject(options)) {
    throw new RegistryError('loader registration options must be an object')
  }
  for (const key of Object.keys(options)) {
    if (key !== 'acceptsContext' && key !== 'sync') {
      throw new RegistryError(`Unknown loader registration option "${key}"`)
    }
  }
  if (options.acceptsContext !== undefined && typeof options.acceptsContext !== 'boolean') {
    throw new RegistryError('acceptsContext must be a boolean')
  }
  if (options.sync !== undefined && typeof options.sync !== 'boolean') {
    throw new RegistryError('sync must be a boolean')
  }
  registry.set(typeId, {
    loaderFn,
    acceptsContext: options.acceptsContext === true,
    sync: options.sync === true
  })
}

async function load(bytes, options = {}) {
  const { manifest, toc, blobs } = validateBundle(bytes)
  const { typeId } = manifest

  if (!typeId) {
    throw new RegistryError('Bundle manifest missing typeId')
  }

  const registration = requireRegistration(typeId)

  assertRequiredLoaders(manifest)
  const context = normalizeLoadContext(options)
  return await invokeLoader(registration, manifest, toc, blobs, context)
}

function loadSync(bytes, options = {}) {
  const { manifest, toc, blobs } = validateBundle(bytes)
  const { typeId } = manifest

  if (!typeId) {
    throw new RegistryError('Bundle manifest missing typeId')
  }

  const registration = requireRegistration(typeId)

  assertRequiredLoaders(manifest)
  if (!registration.sync) {
    throw new RegistryError(
      `Loader for "${typeId}" is not registered as synchronous. ` +
      'Use async load() instead of loadSync().'
    )
  }

  const context = normalizeLoadContext(options)
  const result = invokeLoader(registration, manifest, toc, blobs, context)
  if (result && typeof result.then === 'function') {
    throw new RegistryError(
      `Synchronous loader contract violation for "${typeId}": loader returned a Promise.`
    )
  }
  return result
}

function getRegistry() {
  return new Map([...registry].map(([typeId, registration]) => [
    typeId, registration.loaderFn
  ]))
}

function assertRequiredLoaders(manifest) {
  for (const typeId of manifest.requires || []) {
    if (!registry.has(typeId)) {
      const guidance = typeId === 'wlearn.preprocess.tabular@1'
        ? 'Install @wlearn/preprocess and call await registerPreprocess() before loading.'
        : 'Install and import the corresponding @wlearn/* package before loading this bundle.'
      throw new RegistryError(
        `Missing required loader for nested typeId "${typeId}". ` +
        guidance
      )
    }
  }
}

function requireRegistration(typeId) {
  const registration = registry.get(typeId)
  if (registration) return registration
  const available = [...registry.keys()]
  const list = available.length > 0
    ? `Registered loaders: ${available.join(', ')}`
    : 'No loaders registered'
  const guidance = typeId === 'wlearn.preprocess.tabular@1'
    ? 'Install @wlearn/preprocess and call await registerPreprocess() before loading.'
    : 'Install the corresponding @wlearn/* package and import it to register the loader.'
  throw new RegistryError(
    `No loader registered for typeId "${typeId}". ${list}. ${guidance}`
  )
}

function invokeLoader(registration, manifest, toc, blobs, context) {
  return registration.acceptsContext
    ? registration.loaderFn(manifest, toc, blobs, context)
    : registration.loaderFn(manifest, toc, blobs)
}

function normalizeLoadContext(options) {
  if (options && options[LOAD_CONTEXT] === true) return options
  if (!isPlainObject(options)) {
    throw new RegistryError('load options must be an object')
  }
  for (const key of Object.keys(options)) {
    if (key !== 'loaderOptions') {
      throw new RegistryError(`Unknown load option "${key}"`)
    }
  }
  const loaderOptions = options.loaderOptions === undefined
    ? {}
    : options.loaderOptions
  if (!isPlainObject(loaderOptions)) {
    throw new RegistryError('loaderOptions must be an object keyed by typeId')
  }
  const seen = new Set()
  const snapshot = Object.fromEntries(Object.entries(loaderOptions).map(
    ([typeId, value]) => [
      typeId,
      snapshotLoadOption(value, `loaderOptions["${typeId}"]`, seen),
    ]
  ))
  const context = {
    loaderOptions: Object.freeze(snapshot)
  }
  Object.defineProperty(context, LOAD_CONTEXT, { value: true })
  return Object.freeze(context)
}

function snapshotLoadOption(value, label, seen) {
  if (Array.isArray(value)) {
    if (seen.has(value)) throw new RegistryError(`${label} must not contain cycles`)
    seen.add(value)
    const copy = value.map((item, index) =>
      snapshotLoadOption(item, `${label}[${index}]`, seen)
    )
    seen.delete(value)
    return Object.freeze(copy)
  }
  if (!isPlainObject(value)) return value
  if (seen.has(value)) throw new RegistryError(`${label} must not contain cycles`)
  seen.add(value)
  const copy = Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key, snapshotLoadOption(item, `${label}.${key}`, seen),
  ]))
  seen.delete(value)
  return Object.freeze(copy)
}

function isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

module.exports = { register, load, loadSync, getRegistry, assertRequiredLoaders }
