const { RegistryError } = require('./errors.js')
const { decodeBundle } = require('./bundle.js')

const registry = new Map()

function register(typeId, loaderFn) {
  if (typeof typeId !== 'string' || !typeId.includes('@')) {
    throw new RegistryError(`Invalid typeId "${typeId}": must contain "@" (e.g. "wlearn.liblinear.classifier@1")`)
  }
  if (typeof loaderFn !== 'function') {
    throw new RegistryError('loaderFn must be a function')
  }
  registry.set(typeId, loaderFn)
}

async function load(bytes) {
  const { manifest, toc, blobs } = decodeBundle(bytes)
  const { typeId } = manifest

  if (!typeId) {
    throw new RegistryError('Bundle manifest missing typeId')
  }

  const loaderFn = registry.get(typeId)
  if (!loaderFn) {
    const available = [...registry.keys()]
    const list = available.length > 0
      ? `Registered loaders: ${available.join(', ')}`
      : 'No loaders registered'
    throw new RegistryError(
      `No loader registered for typeId "${typeId}". ${list}. ` +
      `Install the corresponding @wlearn/* package and import it to register the loader.`
    )
  }

  return await loaderFn(manifest, toc, blobs)
}

function loadSync(bytes) {
  const { manifest, toc, blobs } = decodeBundle(bytes)
  const { typeId } = manifest

  if (!typeId) {
    throw new RegistryError('Bundle manifest missing typeId')
  }

  const loaderFn = registry.get(typeId)
  if (!loaderFn) {
    const available = [...registry.keys()]
    const list = available.length > 0
      ? `Registered loaders: ${available.join(', ')}`
      : 'No loaders registered'
    throw new RegistryError(
      `No loader registered for typeId "${typeId}". ${list}. ` +
      `Install the corresponding @wlearn/* package and import it to register the loader.`
    )
  }

  const result = loaderFn(manifest, toc, blobs)
  if (result && typeof result.then === 'function') {
    throw new RegistryError(
      `Loader for "${typeId}" returned a Promise. Use async load() instead of loadSync().`
    )
  }
  return result
}

function getRegistry() {
  return new Map(registry)
}

module.exports = { register, load, loadSync, getRegistry }
