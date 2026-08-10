const { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE } = require('@wlearn/types')
const { BundleError } = require('./errors.js')
const { sha256Sync } = require('./hash.js')

const DEFAULT_BUNDLE_LIMITS = Object.freeze({
  maxBundleBytes: 1024 * 1024 * 1024,
  maxManifestBytes: 16 * 1024 * 1024,
  maxTocBytes: 16 * 1024 * 1024,
  maxArtifacts: 10000,
  maxArtifactBytes: 1024 * 1024 * 1024,
  maxNestingDepth: 32,
  maxDecodedBytes: 4 * 1024 * 1024 * 1024,
})

const TYPE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*@[0-9]+$/
const SHA256_RE = /^[0-9a-f]{64}$/
const TOC_ENTRY_KEYS = Object.freeze(['id', 'offset', 'length', 'sha256', 'mediaType'])
const ARTIFACT_DECLARATION_KEYS = Object.freeze(['id', 'length', 'sha256', 'mediaType'])

function _limits(options = {}) {
  const overrides = options.limits || options
  const limits = { ...DEFAULT_BUNDLE_LIMITS }
  for (const name of Object.keys(DEFAULT_BUNDLE_LIMITS)) {
    if (Object.prototype.hasOwnProperty.call(overrides, name)) limits[name] = overrides[name]
  }
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new BundleError(`Invalid bundle limit ${name}: ${value}`)
    }
  }
  return limits
}

function _validateTypeId(typeId, field = 'manifest.typeId') {
  if (typeof typeId !== 'string' || !TYPE_ID_RE.test(typeId)) {
    throw new BundleError(`${field} must be a versioned typeId such as "wlearn.model@1"`)
  }
}

function _validateArtifactId(id, field = 'artifact.id') {
  if (typeof id !== 'string' || id.length === 0 || id.length > 1024) {
    throw new BundleError(`${field} must be a non-empty string of at most 1024 characters`)
  }
}

function _validateJsonValue(value, path = 'value', ancestors = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new BundleError(`${path} must contain only finite numbers`)
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new BundleError(`${path} contains an integer outside the portable safe range`)
    }
    return
  }
  if (typeof value !== 'object') {
    throw new BundleError(`${path} contains a value that is not representable in JSON`)
  }
  if (ancestors.has(value)) throw new BundleError(`${path} contains a circular reference`)
  ancestors.add(value)
  try {
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) _validateJsonValue(value[i], `${path}[${i}]`, ancestors)
      return
    }
    const prototype = Object.getPrototypeOf(value)
    if (prototype !== Object.prototype && prototype !== null) {
      throw new BundleError(`${path} must contain only plain JSON objects`)
    }
    for (const [key, child] of Object.entries(value)) {
      _validateJsonValue(child, `${path}.${key}`, ancestors)
    }
  } finally {
    ancestors.delete(value)
  }
}

function _validateExactKeys(value, expectedKeys, path) {
  const actualKeys = Object.keys(value).sort()
  const canonicalKeys = [...expectedKeys].sort()
  if (actualKeys.length !== canonicalKeys.length ||
      actualKeys.some((key, index) => key !== canonicalKeys[index])) {
    throw new BundleError(
      `${path} must contain exactly: ${canonicalKeys.join(', ')}`
    )
  }
}

// Deterministic JSON: sorted keys recursively, no whitespace, array order preserved
function stableStringify(val) {
  if (val === null || val === undefined) return JSON.stringify(val)
  if (typeof val !== 'object') return JSON.stringify(val)
  if (Array.isArray(val)) {
    return '[' + val.map(v => stableStringify(v)).join(',') + ']'
  }
  const keys = Object.keys(val).sort()
  return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(val[k])).join(',') + '}'
}

const textEncoder = new TextEncoder()
const textDecoder = new TextDecoder('utf-8', { fatal: true })

/**
 * Encode a wlearn bundle (WLRN format).
 *
 * @param {Object} manifest - Bundle manifest. Must include `typeId` (e.g. `'wlearn.xgboost.classifier@1'`).
 *   May include `params`, `requires`, `seed`, or any model-specific metadata.
 * @param {Array<{id: string, mediaType?: string, data: Uint8Array}>} artifacts - Artifact blobs.
 *   Each entry has `id` (unique within the bundle), optional `mediaType`, and `data` (raw bytes).
 *   Artifacts are sorted by `id` for determinism; SHA-256 hashes are computed automatically.
 * @returns {Uint8Array} The encoded bundle bytes (header + manifest JSON + TOC JSON + blob region).
 */
function encodeBundle(manifest, artifacts, options = {}) {
  const limits = _limits(options)
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw new BundleError('manifest must be an object')
  }
  _validateTypeId(manifest.typeId)
  if (manifest.params !== undefined &&
      (!manifest.params || typeof manifest.params !== 'object' || Array.isArray(manifest.params))) {
    throw new BundleError('manifest.params must be an object')
  }
  if (manifest.requires !== undefined && !Array.isArray(manifest.requires)) {
    throw new BundleError('manifest.requires must be an array')
  }
  if (!Array.isArray(artifacts)) throw new BundleError('artifacts must be an array')
  if (artifacts.length > limits.maxArtifacts) {
    throw new BundleError(`Too many artifacts: ${artifacts.length} (maximum ${limits.maxArtifacts})`)
  }

  for (const art of artifacts) {
    if (!art || typeof art !== 'object' || Array.isArray(art)) {
      throw new BundleError('artifact must be an object')
    }
    _validateArtifactId(art.id)
  }

  // Sort artifacts by id for determinism
  const sorted = [...artifacts].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)

  // Build TOC and compute blob region
  let blobOffset = 0
  const toc = []
  const blobs = []
  const ids = new Set()
  const derivedRequires = new Set(manifest.requires || [])
  const nestedValidationState = { decodedBytes: 0 }

  for (const typeId of derivedRequires) _validateTypeId(typeId, 'manifest.requires entry')

  for (const art of sorted) {
    if (ids.has(art.id)) throw new BundleError(`Duplicate artifact id "${art.id}"`)
    ids.add(art.id)
    if (!(art.data instanceof Uint8Array) && !(art.data instanceof ArrayBuffer)) {
      throw new BundleError(`Artifact "${art.id}" data must be Uint8Array or ArrayBuffer`)
    }
    const data = art.data instanceof Uint8Array ? art.data : new Uint8Array(art.data)
    if (data.length > limits.maxArtifactBytes) {
      throw new BundleError(`Artifact "${art.id}" exceeds maximum size ${limits.maxArtifactBytes}`)
    }
    const hash = sha256Sync(data)
    const mediaType = art.mediaType || 'application/octet-stream'
    if (typeof mediaType !== 'string' || mediaType.length === 0) {
      throw new BundleError(`Artifact "${art.id}" mediaType must be a non-empty string`)
    }
    const entry = { id: art.id, offset: blobOffset, length: data.length, sha256: hash, mediaType }
    toc.push(entry)
    blobs.push(data)
    blobOffset += data.length

    if (mediaType === 'application/x-wlearn-bundle') {
      const nested = validateBundle(
        data,
        { ...options, allowLegacyManifest: false },
        { state: nestedValidationState, depth: 1 }
      ).manifest
      derivedRequires.add(nested.typeId)
      for (const typeId of nested.requires || []) derivedRequires.add(typeId)
    }
  }

  const artifactDeclarations = toc.map(({ id, length, sha256, mediaType }) => ({
    id, length, sha256, mediaType,
  }))
  const fullManifest = {
    ...manifest,
    typeId: manifest.typeId,
    bundleVersion: BUNDLE_VERSION,
    requires: [...derivedRequires].sort(),
    artifacts: artifactDeclarations,
    params: manifest.params || {},
  }
  _validateJsonValue(fullManifest, 'manifest')
  const manifestBytes = textEncoder.encode(stableStringify(fullManifest))
  const tocBytes = textEncoder.encode(stableStringify(toc))

  if (manifestBytes.length > limits.maxManifestBytes) {
    throw new BundleError(`Manifest exceeds maximum size ${limits.maxManifestBytes}`)
  }
  if (tocBytes.length > limits.maxTocBytes) {
    throw new BundleError(`TOC exceeds maximum size ${limits.maxTocBytes}`)
  }

  const totalLen = HEADER_SIZE + manifestBytes.length + tocBytes.length + blobOffset
  if (!Number.isSafeInteger(totalLen) || totalLen > limits.maxBundleBytes) {
    throw new BundleError(`Bundle exceeds maximum size ${limits.maxBundleBytes}`)
  }
  if (nestedValidationState.decodedBytes + totalLen > limits.maxDecodedBytes) {
    throw new BundleError(`Decoded nested bundle bytes exceed maximum ${limits.maxDecodedBytes}`)
  }
  const out = new Uint8Array(totalLen)
  const view = new DataView(out.buffer)

  // Header
  out.set(BUNDLE_MAGIC, 0)
  view.setUint32(4, BUNDLE_VERSION, true)
  view.setUint32(8, manifestBytes.length, true)
  view.setUint32(12, tocBytes.length, true)

  // Manifest + TOC + blobs
  out.set(manifestBytes, HEADER_SIZE)
  out.set(tocBytes, HEADER_SIZE + manifestBytes.length)

  let pos = HEADER_SIZE + manifestBytes.length + tocBytes.length
  for (const blob of blobs) {
    out.set(blob, pos)
    pos += blob.length
  }

  return out
}

/**
 * Decode a wlearn bundle (WLRN format).
 *
 * @param {Uint8Array|ArrayBuffer} bytes - Raw bundle bytes.
 * @returns {{manifest: Object, toc: Array<{id: string, offset: number, length: number, sha256: string, mediaType?: string}>, blobs: Uint8Array}}
 *   `manifest` is the parsed manifest object. `toc` is an array of blob descriptors.
 *   `blobs` is the concatenated blob region -- slice individual blobs with
 *   `blobs.subarray(entry.offset, entry.offset + entry.length)`.
 * @throws {BundleError} On invalid magic, unsupported version, truncated data, or malformed JSON.
 */
function decodeBundle(bytes, options = {}) {
  const limits = _limits(options)
  if (!(bytes instanceof Uint8Array) && !(bytes instanceof ArrayBuffer)) {
    throw new BundleError('Bundle input must be Uint8Array or ArrayBuffer')
  }
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)

  if (buf.length > limits.maxBundleBytes) {
    throw new BundleError(`Bundle exceeds maximum size ${limits.maxBundleBytes}`)
  }

  if (buf.length < HEADER_SIZE) {
    throw new BundleError(`Bundle too small: ${buf.length} bytes (minimum ${HEADER_SIZE})`)
  }

  // Verify magic
  for (let i = 0; i < 4; i++) {
    if (buf[i] !== BUNDLE_MAGIC[i]) {
      throw new BundleError('Invalid bundle magic (expected WLRN)')
    }
  }

  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const version = view.getUint32(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new BundleError(`Unsupported bundle version: ${version} (expected ${BUNDLE_VERSION})`)
  }

  const manifestLen = view.getUint32(8, true)
  const tocLen = view.getUint32(12, true)

  if (manifestLen > limits.maxManifestBytes) {
    throw new BundleError(`Manifest exceeds maximum size ${limits.maxManifestBytes}`)
  }
  if (tocLen > limits.maxTocBytes) {
    throw new BundleError(`TOC exceeds maximum size ${limits.maxTocBytes}`)
  }

  if (HEADER_SIZE + manifestLen + tocLen > buf.length) {
    throw new BundleError(`Bundle truncated: header declares ${HEADER_SIZE + manifestLen + tocLen} bytes but got ${buf.length}`)
  }

  let manifest, toc
  try {
    manifest = JSON.parse(textDecoder.decode(buf.subarray(HEADER_SIZE, HEADER_SIZE + manifestLen)))
  } catch (e) {
    throw new BundleError(`Invalid manifest JSON: ${e.message}`)
  }

  try {
    toc = JSON.parse(textDecoder.decode(buf.subarray(HEADER_SIZE + manifestLen, HEADER_SIZE + manifestLen + tocLen)))
  } catch (e) {
    throw new BundleError(`Invalid TOC JSON: ${e.message}`)
  }

  const blobStart = HEADER_SIZE + manifestLen + tocLen
  const blobRegionLen = buf.length - blobStart

  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw new BundleError('Manifest JSON must be an object')
  }
  _validateTypeId(manifest.typeId)
  if (manifest.bundleVersion !== BUNDLE_VERSION) {
    throw new BundleError(`Manifest bundleVersion must be ${BUNDLE_VERSION}`)
  }
  if (!Array.isArray(toc)) throw new BundleError('TOC JSON must be an array')
  _validateJsonValue(toc, 'toc')
  if (toc.length > limits.maxArtifacts) {
    throw new BundleError(`Too many artifacts: ${toc.length} (maximum ${limits.maxArtifacts})`)
  }
  if (manifest.requires !== undefined) {
    if (!Array.isArray(manifest.requires)) throw new BundleError('manifest.requires must be an array')
    const requires = new Set()
    for (const typeId of manifest.requires) {
      _validateTypeId(typeId, 'manifest.requires entry')
      if (requires.has(typeId)) throw new BundleError(`Duplicate manifest requirement "${typeId}"`)
      requires.add(typeId)
    }
  }
  if (manifest.params !== undefined &&
      (!manifest.params || typeof manifest.params !== 'object' || Array.isArray(manifest.params))) {
    throw new BundleError('manifest.params must be an object')
  }
  if (manifest.seed !== undefined && !Number.isSafeInteger(manifest.seed)) {
    throw new BundleError('manifest.seed must be a safe integer')
  }
  if (manifest.metadata !== undefined &&
      (!manifest.metadata || typeof manifest.metadata !== 'object' || Array.isArray(manifest.metadata))) {
    throw new BundleError('manifest.metadata must be an object')
  }
  _validateJsonValue(manifest, 'manifest')

  // Validate TOC entries: typed fields, unique ids, no overlaps, within bounds.
  const ids = new Set()
  for (let i = 0; i < toc.length; i++) {
    const entry = toc[i]
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new BundleError(`TOC entry ${i} must be an object`)
    }
    if (options.allowLegacyManifest === false) {
      _validateExactKeys(entry, TOC_ENTRY_KEYS, `TOC entry ${i}`)
    }
    _validateArtifactId(entry.id, `TOC entry ${i}.id`)
    if (ids.has(entry.id)) throw new BundleError(`Duplicate artifact id "${entry.id}"`)
    ids.add(entry.id)
    if (!Number.isSafeInteger(entry.offset) || !Number.isSafeInteger(entry.length) ||
        entry.offset < 0 || entry.length < 0 || entry.length > limits.maxArtifactBytes ||
        entry.offset + entry.length > blobRegionLen) {
      throw new BundleError(`TOC entry "${entry.id}" out of bounds: offset=${entry.offset}, length=${entry.length}, blobRegion=${blobRegionLen}`)
    }
    if (typeof entry.sha256 !== 'string' || !SHA256_RE.test(entry.sha256)) {
      throw new BundleError(`TOC entry "${entry.id}" has invalid SHA-256`)
    }
    if (entry.mediaType !== undefined &&
        (typeof entry.mediaType !== 'string' || entry.mediaType.length === 0)) {
      throw new BundleError(`TOC entry "${entry.id}" has invalid mediaType`)
    }
  }

  const byOffset = [...toc].sort((a, b) => a.offset - b.offset || a.length - b.length)
  let coveredBytes = 0
  for (const entry of byOffset) {
    if (entry.offset < coveredBytes) {
      throw new BundleError(`TOC entry "${entry.id}" overlaps a previous artifact`)
    }
    if (entry.offset > coveredBytes) {
      throw new BundleError(`Unreferenced blob gap before artifact "${entry.id}"`)
    }
    coveredBytes = entry.offset + entry.length
  }
  if (coveredBytes !== blobRegionLen) {
    throw new BundleError(`Unreferenced trailing blob bytes: ${blobRegionLen - coveredBytes}`)
  }

  const strictManifest = options.allowLegacyManifest === false
  if (manifest.artifacts !== undefined) {
    if (!Array.isArray(manifest.artifacts)) throw new BundleError('manifest.artifacts must be an array')
    if (manifest.artifacts.length !== toc.length) {
      throw new BundleError('manifest.artifacts length does not match TOC')
    }
    for (let i = 0; i < toc.length; i++) {
      const declared = manifest.artifacts[i]
      const entry = toc[i]
      if (strictManifest && declared && typeof declared === 'object' && !Array.isArray(declared)) {
        _validateExactKeys(
          declared, ARTIFACT_DECLARATION_KEYS,
          `manifest.artifacts[${i}]`
        )
      }
      if (!declared || declared.id !== entry.id || declared.length !== entry.length ||
          declared.sha256 !== entry.sha256 || declared.mediaType !== entry.mediaType) {
        throw new BundleError(`manifest artifact declaration does not match TOC entry "${entry.id}"`)
      }
    }
  } else if (strictManifest) {
    throw new BundleError('manifest.artifacts is required')
  }

  if (strictManifest) {
    if (!Object.prototype.hasOwnProperty.call(manifest, 'requires')) {
      throw new BundleError('manifest.requires is required')
    }
    if (!Object.prototype.hasOwnProperty.call(manifest, 'params')) {
      throw new BundleError('manifest.params is required')
    }
    for (let i = 0; i < toc.length; i++) {
      if (toc[i].mediaType === undefined) {
        throw new BundleError(`TOC entry "${toc[i].id}" mediaType is required`)
      }
      if (i > 0 && toc[i - 1].id >= toc[i].id) {
        throw new BundleError('TOC entries must be ordered by unique artifact id')
      }
    }
  }

  const blobs = buf.subarray(blobStart)
  return { manifest, toc, blobs }
}

/**
 * Encode a value as deterministic JSON bytes (sorted keys, no whitespace).
 * @param {*} obj - Value to serialize.
 * @returns {Uint8Array} UTF-8 encoded JSON bytes.
 */
function encodeJSON(obj) {
  _validateJsonValue(obj)
  return textEncoder.encode(stableStringify(obj))
}

/**
 * Decode UTF-8 JSON bytes to a JS value.
 * @param {Uint8Array|ArrayBuffer} bytes - UTF-8 encoded JSON.
 * @returns {*} Parsed value.
 */
function decodeJSON(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  return JSON.parse(textDecoder.decode(buf))
}

/**
 * Decode and validate a bundle, verifying SHA-256 hashes of all blobs.
 * @param {Uint8Array|ArrayBuffer} bytes - Raw bundle bytes.
 * @returns {{manifest: Object, toc: Array, blobs: Uint8Array}} Same shape as `decodeBundle`.
 * @throws {BundleError} On format errors or hash mismatch.
 */
function validateBundle(bytes, options = {}, context = null) {
  const { manifest, toc, blobs } = decodeBundle(bytes, options)
  const limits = _limits(options)
  const state = context?.state || { decodedBytes: 0 }
  const depth = context?.depth || 0
  if (depth > limits.maxNestingDepth) {
    throw new BundleError(`Nested bundle depth ${depth} exceeds maximum ${limits.maxNestingDepth}`)
  }
  const inputLength = bytes instanceof Uint8Array ? bytes.byteLength : bytes.byteLength
  state.decodedBytes += inputLength
  if (state.decodedBytes > limits.maxDecodedBytes) {
    throw new BundleError(`Decoded nested bundle bytes exceed maximum ${limits.maxDecodedBytes}`)
  }

  const nestedRequires = new Set()

  for (const entry of toc) {
    const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
    const hash = sha256Sync(blob)
    if (hash !== entry.sha256) {
      throw new BundleError(`SHA-256 mismatch for "${entry.id}": expected ${entry.sha256}, got ${hash}`)
    }
    if (entry.mediaType === 'application/x-wlearn-bundle') {
      const nested = validateBundle(blob, options, { state, depth: depth + 1 })
      nestedRequires.add(nested.manifest.typeId)
      for (const typeId of nested.manifest.requires || []) nestedRequires.add(typeId)
    }
  }

  if (manifest.requires !== undefined) {
    const declared = new Set(manifest.requires)
    for (const typeId of nestedRequires) {
      if (!declared.has(typeId)) {
        throw new BundleError(`manifest.requires is missing nested loader "${typeId}"`)
      }
    }
  }

  return { manifest, toc, blobs }
}

module.exports = {
  DEFAULT_BUNDLE_LIMITS,
  encodeBundle,
  decodeBundle,
  encodeJSON,
  decodeJSON,
  validateBundle,
}
