import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { encodeBundle, decodeBundle, validateBundle } from '../src/bundle.js'
import { BundleError } from '../src/errors.js'
import { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE } from '@wlearn/types'

describe('encodeBundle + decodeBundle', () => {
  it('round-trips manifest and artifacts', () => {
    const manifest = { typeId: 'wlearn.test@1', params: { lr: 0.1 } }
    const artifacts = [
      { id: 'model', data: new Uint8Array([1, 2, 3, 4]) },
      { id: 'config', data: new Uint8Array([10, 20]), mediaType: 'application/json' }
    ]

    const bytes = encodeBundle(manifest, artifacts)
    const { manifest: m, toc, blobs } = decodeBundle(bytes)

    assert.equal(m.typeId, 'wlearn.test@1')
    assert.equal(m.bundleVersion, BUNDLE_VERSION)
    assert.deepEqual(m.params, { lr: 0.1 })
    assert.equal(toc.length, 2)

    // Artifacts sorted by id: config before model
    assert.equal(toc[0].id, 'config')
    assert.equal(toc[1].id, 'model')

    const configBlob = blobs.subarray(toc[0].offset, toc[0].offset + toc[0].length)
    assert.deepEqual([...configBlob], [10, 20])
    assert.equal(toc[0].mediaType, 'application/json')

    const modelBlob = blobs.subarray(toc[1].offset, toc[1].offset + toc[1].length)
    assert.deepEqual([...modelBlob], [1, 2, 3, 4])
  })

  it('handles empty artifacts', () => {
    const manifest = { typeId: 'wlearn.empty@1' }
    const bytes = encodeBundle(manifest, [])
    const { manifest: m, toc } = decodeBundle(bytes)
    assert.equal(m.typeId, 'wlearn.empty@1')
    assert.equal(toc.length, 0)
  })

  it('preserves header magic and version', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    assert.equal(bytes[0], 0x57) // W
    assert.equal(bytes[1], 0x4c) // L
    assert.equal(bytes[2], 0x52) // R
    assert.equal(bytes[3], 0x4e) // N

    const view = new DataView(bytes.buffer)
    assert.equal(view.getUint32(4, true), BUNDLE_VERSION)
  })

  it('produces deterministic output', () => {
    const manifest = { typeId: 'wlearn.det@1', params: { b: 2, a: 1 } }
    const artifacts = [
      { id: 'z', data: new Uint8Array([1]) },
      { id: 'a', data: new Uint8Array([2]) }
    ]
    const a = encodeBundle(manifest, artifacts)
    const b = encodeBundle(manifest, artifacts)
    assert.deepEqual(a, b)
  })
})

describe('validateBundle', () => {
  it('passes on valid bundle', () => {
    const manifest = { typeId: 'wlearn.test@1' }
    const artifacts = [{ id: 'data', data: new Uint8Array([42, 43, 44]) }]
    const bytes = encodeBundle(manifest, artifacts)
    const { manifest: m, toc } = validateBundle(bytes)
    assert.equal(m.typeId, 'wlearn.test@1')
    assert.equal(toc.length, 1)
  })

  it('throws on corrupted blob', () => {
    const bytes = encodeBundle(
      { typeId: 'wlearn.test@1' },
      [{ id: 'data', data: new Uint8Array([1, 2, 3]) }]
    )
    // Corrupt the last byte (inside blob region)
    bytes[bytes.length - 1] ^= 0xff
    assert.throws(() => validateBundle(bytes), BundleError)
  })
})

describe('decodeBundle validation', () => {
  it('rejects truncated header', () => {
    assert.throws(() => decodeBundle(new Uint8Array(10)), BundleError)
  })

  it('rejects bad magic', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    bytes[0] = 0x00
    assert.throws(() => decodeBundle(bytes), BundleError)
  })

  it('rejects bad version', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    const view = new DataView(bytes.buffer)
    view.setUint32(4, 99, true)
    assert.throws(() => decodeBundle(bytes), (err) => {
      assert(err instanceof BundleError)
      assert(err.message.includes('99'))
      return true
    })
  })

  it('rejects when manifestLen + tocLen exceeds bytes', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    const view = new DataView(bytes.buffer)
    view.setUint32(8, 999999, true) // set manifestLen way too large
    assert.throws(() => decodeBundle(bytes), BundleError)
  })

  it('rejects overlapping TOC entries', () => {
    // Build a valid bundle then manually craft overlapping TOC
    // This is hard to trigger via encodeBundle, so we construct manually
    const manifest = JSON.stringify({ typeId: 'wlearn.test@1', bundleVersion: 1 })
    const toc = JSON.stringify([
      { id: 'a', offset: 0, length: 10, sha256: 'abc' },
      { id: 'b', offset: 5, length: 10, sha256: 'def' }
    ])
    const enc = new TextEncoder()
    const mBytes = enc.encode(manifest)
    const tBytes = enc.encode(toc)
    const blob = new Uint8Array(20)

    const total = HEADER_SIZE + mBytes.length + tBytes.length + blob.length
    const buf = new Uint8Array(total)
    const view = new DataView(buf.buffer)
    buf.set(BUNDLE_MAGIC, 0)
    view.setUint32(4, 1, true)
    view.setUint32(8, mBytes.length, true)
    view.setUint32(12, tBytes.length, true)
    buf.set(mBytes, HEADER_SIZE)
    buf.set(tBytes, HEADER_SIZE + mBytes.length)
    buf.set(blob, HEADER_SIZE + mBytes.length + tBytes.length)

    assert.throws(() => decodeBundle(buf), BundleError)
  })

  it('rejects TOC entry out of bounds', () => {
    const manifest = JSON.stringify({ typeId: 'wlearn.test@1', bundleVersion: 1 })
    const toc = JSON.stringify([
      { id: 'a', offset: 0, length: 100, sha256: 'abc' }
    ])
    const enc = new TextEncoder()
    const mBytes = enc.encode(manifest)
    const tBytes = enc.encode(toc)
    const blob = new Uint8Array(5) // only 5 bytes, but TOC says 100

    const total = HEADER_SIZE + mBytes.length + tBytes.length + blob.length
    const buf = new Uint8Array(total)
    const view = new DataView(buf.buffer)
    buf.set(BUNDLE_MAGIC, 0)
    view.setUint32(4, 1, true)
    view.setUint32(8, mBytes.length, true)
    view.setUint32(12, tBytes.length, true)
    buf.set(mBytes, HEADER_SIZE)
    buf.set(tBytes, HEADER_SIZE + mBytes.length)
    buf.set(blob, HEADER_SIZE + mBytes.length + tBytes.length)

    assert.throws(() => decodeBundle(buf), BundleError)
  })
})
