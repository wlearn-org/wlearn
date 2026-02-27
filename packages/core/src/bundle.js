import { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE } from '@wlearn/types'
import { BundleError } from './errors.js'
import { sha256Sync } from './hash.js'

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
const textDecoder = new TextDecoder()

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
export function encodeBundle(manifest, artifacts) {
  // Sort artifacts by id for determinism
  const sorted = [...artifacts].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)

  // Build TOC and compute blob region
  let blobOffset = 0
  const toc = []
  const blobs = []

  for (const art of sorted) {
    const data = art.data instanceof Uint8Array ? art.data : new Uint8Array(art.data)
    const hash = sha256Sync(data)
    const entry = { id: art.id, offset: blobOffset, length: data.length, sha256: hash }
    if (art.mediaType) entry.mediaType = art.mediaType
    toc.push(entry)
    blobs.push(data)
    blobOffset += data.length
  }

  const fullManifest = { ...manifest, bundleVersion: BUNDLE_VERSION }
  const manifestBytes = textEncoder.encode(stableStringify(fullManifest))
  const tocBytes = textEncoder.encode(stableStringify(toc))

  const totalLen = HEADER_SIZE + manifestBytes.length + tocBytes.length + blobOffset
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
export function decodeBundle(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)

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

  // Validate TOC entries: no overlaps, within bounds
  for (let i = 0; i < toc.length; i++) {
    const entry = toc[i]
    if (entry.offset < 0 || entry.length < 0 || entry.offset + entry.length > blobRegionLen) {
      throw new BundleError(`TOC entry "${entry.id}" out of bounds: offset=${entry.offset}, length=${entry.length}, blobRegion=${blobRegionLen}`)
    }
    for (let j = i + 1; j < toc.length; j++) {
      const other = toc[j]
      const aStart = entry.offset, aEnd = entry.offset + entry.length
      const bStart = other.offset, bEnd = other.offset + other.length
      if (aStart < bEnd && bStart < aEnd && entry.length > 0 && other.length > 0) {
        throw new BundleError(`TOC entries "${entry.id}" and "${other.id}" overlap`)
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
export function encodeJSON(obj) {
  return textEncoder.encode(stableStringify(obj))
}

/**
 * Decode UTF-8 JSON bytes to a JS value.
 * @param {Uint8Array|ArrayBuffer} bytes - UTF-8 encoded JSON.
 * @returns {*} Parsed value.
 */
export function decodeJSON(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  return JSON.parse(textDecoder.decode(buf))
}

/**
 * Decode and validate a bundle, verifying SHA-256 hashes of all blobs.
 * @param {Uint8Array|ArrayBuffer} bytes - Raw bundle bytes.
 * @returns {{manifest: Object, toc: Array, blobs: Uint8Array}} Same shape as `decodeBundle`.
 * @throws {BundleError} On format errors or hash mismatch.
 */
export function validateBundle(bytes) {
  const { manifest, toc, blobs } = decodeBundle(bytes)

  for (const entry of toc) {
    const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
    const hash = sha256Sync(blob)
    if (hash !== entry.sha256) {
      throw new BundleError(`SHA-256 mismatch for "${entry.id}": expected ${entry.sha256}, got ${hash}`)
    }
  }

  return { manifest, toc, blobs }
}
