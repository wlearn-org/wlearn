const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  encodeBundle, decodeBundle, validateBundle, encodeJSON, DEFAULT_BUNDLE_LIMITS
} = require('../src/bundle.js')
const { sha256Sync } = require('../src/hash.js')
const { BundleError } = require('../src/errors.js')
const { BUNDLE_MAGIC, BUNDLE_VERSION, HEADER_SIZE } = require('@wlearn/types')

// Exact published-era fixture bytes from git 96a57f9:
// fixtures/liblinear-classifier.wlrn. Keep immutable as a compatibility lock.
const HISTORICAL_PRE_CANONICAL_BUNDLE = Uint8Array.from(Buffer.from(
  'V0xSTgEAAABjAAAAdAAAAHsiYnVuZGxlVmVyc2lvbiI6MSwicGFyYW1zIjp7IkMiOjEsImVwcyI6MC4wMSwic29sdmVyIjowfSwidHlwZUlkIjoid2xlYXJuLmxpYmxpbmVhci5jbGFzc2lmaWVyQDEifVt7ImlkIjoibW9kZWwiLCJsZW5ndGgiOjEwNywib2Zmc2V0IjowLCJzaGEyNTYiOiJhMDdiZDAwNDI5M2Y5OTUxZjY2MjI5OWUyMDU5ZTdhYThhNDNmNWI1ZWQ4ZGMyNDIwN2U4MGU1YzUyNzg5ZTA1In1dc29sdmVyX3R5cGUgTDJSX0xSCm5yX2NsYXNzIDIKbGFiZWwgMCAxCm5yX2ZlYXR1cmUgMgpiaWFzIC0xCncKLTAuODczNjI4NzEwMTc2MTExOTkgCi0wLjk1NTUzNzYxNTE4MDUyMDA1IAo=',
  'base64'
))

function makeRawBundle(manifest, toc = [], blob = new Uint8Array(0), manifestBytes = null) {
  const enc = new TextEncoder()
  const mBytes = manifestBytes || enc.encode(JSON.stringify(manifest))
  const tBytes = enc.encode(JSON.stringify(toc))
  const buf = new Uint8Array(HEADER_SIZE + mBytes.length + tBytes.length + blob.length)
  const view = new DataView(buf.buffer)
  buf.set(BUNDLE_MAGIC, 0)
  view.setUint32(4, BUNDLE_VERSION, true)
  view.setUint32(8, mBytes.length, true)
  view.setUint32(12, tBytes.length, true)
  buf.set(mBytes, HEADER_SIZE)
  buf.set(tBytes, HEADER_SIZE + mBytes.length)
  buf.set(blob, HEADER_SIZE + mBytes.length + tBytes.length)
  return buf
}

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
    assert.deepEqual(m.requires, [])
    assert.deepEqual(m.artifacts, toc.map(({ id, length, sha256, mediaType }) => ({
      id, length, sha256, mediaType
    })))
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

  it('derives nested loader requirements', () => {
    const nested = encodeBundle({
      typeId: 'wlearn.child@1',
      requires: ['wlearn.dependency@1']
    }, [])
    const bytes = encodeBundle({ typeId: 'wlearn.parent@1' }, [{
      id: 'child', mediaType: 'application/x-wlearn-bundle', data: nested
    }])
    assert.deepEqual(decodeBundle(bytes).manifest.requires, [
      'wlearn.child@1', 'wlearn.dependency@1'
    ])
    assert.doesNotThrow(() => validateBundle(bytes, { allowLegacyManifest: false }))
  })

  it('rejects legacy nested bundles instead of emitting a non-canonical parent', () => {
    const legacy = makeRawBundle({
      typeId: 'wlearn.legacy-child@1', bundleVersion: 1
    })
    assert.throws(() => encodeBundle({ typeId: 'wlearn.parent@1' }, [{
      id: 'child', mediaType: 'application/x-wlearn-bundle', data: legacy
    }]), /manifest.artifacts is required/)
  })

  it('rejects malformed writer inputs', () => {
    assert.throws(() => encodeBundle({}, []), BundleError)
    assert.throws(() => encodeBundle({ typeId: 'unversioned' }, []), BundleError)
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1', params: [] }, []), BundleError)
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1', requires: 'bad' }, []), BundleError)
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1' }, [null]), BundleError)
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1' }, [
      { id: 'same', data: new Uint8Array(0) },
      { id: 'same', data: new Uint8Array(0) }
    ]), /Duplicate artifact/)
    assert.throws(() => encodeBundle({
      typeId: 'wlearn.test@1', metadata: { value: undefined }
    }, []), /not representable in JSON/)
    assert.throws(() => encodeBundle({
      typeId: 'wlearn.test@1', metadata: { value: Number.NaN }
    }, []), /finite/)
    assert.throws(() => encodeBundle({
      typeId: 'wlearn.test@1', metadata: { value: Number.MAX_SAFE_INTEGER + 1 }
    }, []), /safe range/)
    const circular = {}
    circular.self = circular
    assert.throws(() => encodeJSON(circular), /circular/)
  })

  it('enforces configurable writer limits', () => {
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1' }, [
      { id: 'a', data: new Uint8Array(0) }
    ], { maxArtifacts: 0 }), /Too many artifacts/)
    assert.throws(() => encodeBundle({ typeId: 'wlearn.test@1' }, [
      { id: 'a', data: new Uint8Array(2) }
    ], { limits: { maxArtifactBytes: 1 } }), /maximum size/)
    const child = encodeBundle({ typeId: 'wlearn.child@1' }, [])
    const nestedArtifact = [{
      id: 'child', data: child, mediaType: 'application/x-wlearn-bundle'
    }]
    assert.throws(() => encodeBundle(
      { typeId: 'wlearn.parent@1' }, nestedArtifact,
      { maxNestingDepth: 0 }
    ), /depth/)
    assert.throws(() => encodeBundle(
      { typeId: 'wlearn.parent@1' }, nestedArtifact,
      { maxDecodedBytes: child.length }
    ), /Decoded nested bundle bytes/)
    assert(Object.isFrozen(DEFAULT_BUNDLE_LIMITS))
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
    const canonical = encodeBundle(
      { typeId: 'wlearn.test@1' },
      [{ id: 'data', data: new Uint8Array([1, 2, 3]) }]
    )
    for (const options of [{}, { allowLegacyManifest: false }]) {
      const corrupted = canonical.slice()
      corrupted[corrupted.length - 1] ^= 0xff
      assert.throws(() => validateBundle(corrupted, options), BundleError)
    }
  })

  it('recursively validates nested bundle hashes', () => {
    const child = encodeBundle({ typeId: 'wlearn.child@1' }, [
      { id: 'data', data: new Uint8Array([1, 2, 3]) }
    ])
    child[child.length - 1] ^= 0xff
    const childHash = sha256Sync(child)
    const toc = [{
      id: 'child', offset: 0, length: child.length, sha256: childHash,
      mediaType: 'application/x-wlearn-bundle'
    }]
    const manifest = {
      typeId: 'wlearn.parent@1', bundleVersion: 1,
      requires: ['wlearn.child@1'], params: {},
      artifacts: [{
        id: 'child', length: child.length, sha256: childHash,
        mediaType: 'application/x-wlearn-bundle'
      }]
    }
    const parent = makeRawBundle(manifest, toc, child)
    assert.throws(() => validateBundle(parent), /SHA-256 mismatch for "data"/)
  })

  it('enforces nesting depth and cumulative decoded-byte budgets', () => {
    const child = encodeBundle({ typeId: 'wlearn.child@1' }, [])
    const parent = encodeBundle({ typeId: 'wlearn.parent@1' }, [{
      id: 'child', data: child, mediaType: 'application/x-wlearn-bundle'
    }])
    assert.throws(() => validateBundle(parent, { maxNestingDepth: 0 }), /depth/)
    assert.throws(() => validateBundle(parent, {
      maxDecodedBytes: parent.length + child.length - 1
    }), /Decoded nested bundle bytes/)
  })

  it('requires nested loader declarations', () => {
    const child = encodeBundle({ typeId: 'wlearn.child@1' }, [])
    const childHash = sha256Sync(child)
    const toc = [{
      id: 'child', offset: 0, length: child.length, sha256: childHash,
      mediaType: 'application/x-wlearn-bundle'
    }]
    const manifest = {
      typeId: 'wlearn.parent@1', bundleVersion: 1, requires: [], params: {},
      artifacts: [{
        id: 'child', length: child.length, sha256: childHash,
        mediaType: 'application/x-wlearn-bundle'
      }]
    }
    assert.throws(() => validateBundle(makeRawBundle(manifest, toc, child), {
      allowLegacyManifest: false
    }), /missing nested loader/)
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
    const buf = makeRawBundle(
      { typeId: 'wlearn.test@1', bundleVersion: 1 },
      [
        { id: 'a', offset: 0, length: 10, sha256: 'a'.repeat(64) },
        { id: 'b', offset: 5, length: 10, sha256: 'b'.repeat(64) }
      ],
      new Uint8Array(20)
    )
    assert.throws(() => decodeBundle(buf), /overlap/)
  })

  it('rejects TOC entry out of bounds', () => {
    const buf = makeRawBundle(
      { typeId: 'wlearn.test@1', bundleVersion: 1 },
      [{ id: 'a', offset: 0, length: 100, sha256: 'a'.repeat(64) }],
      new Uint8Array(5)
    )
    assert.throws(() => decodeBundle(buf), /out of bounds/)
  })

  it('rejects duplicate ids and malformed TOC fields', () => {
    const manifest = { typeId: 'wlearn.test@1', bundleVersion: 1 }
    const valid = { id: 'a', offset: 0, length: 0, sha256: 'a'.repeat(64) }
    assert.throws(() => decodeBundle(makeRawBundle(manifest, [valid, valid])), /Duplicate/)
    assert.throws(() => decodeBundle(makeRawBundle(manifest, [
      { ...valid, offset: 0.5 }
    ])), /out of bounds/)
    assert.throws(() => decodeBundle(makeRawBundle(manifest, [
      { ...valid, sha256: 'not-a-hash' }
    ])), /invalid SHA-256/)
  })

  it('rejects non-UTF-8 manifest bytes', () => {
    assert.throws(() => decodeBundle(makeRawBundle(null, [], new Uint8Array(0),
      new Uint8Array([0xff]))), /Invalid manifest JSON/)
  })

  it('checks manifest artifact declarations against the TOC', () => {
    const entry = { id: 'a', offset: 0, length: 0, sha256: 'a'.repeat(64), mediaType: 'x/test' }
    const declaration = { id: 'a', length: 1, sha256: entry.sha256, mediaType: entry.mediaType }
    const manifest = {
      typeId: 'wlearn.test@1', bundleVersion: 1, artifacts: [declaration]
    }
    assert.throws(() => decodeBundle(makeRawBundle(manifest, [entry])), /does not match/)
  })

  it('supports legacy manifests explicitly and a strict canonical mode', () => {
    const legacy = makeRawBundle({ typeId: 'wlearn.legacy@1', bundleVersion: 1 })
    assert.equal(decodeBundle(legacy).manifest.typeId, 'wlearn.legacy@1')
    assert.throws(
      () => decodeBundle(legacy, { allowLegacyManifest: false }),
      /manifest.artifacts is required/
    )
    const artifactsOnly = makeRawBundle({
      typeId: 'wlearn.incomplete@1', bundleVersion: 1, artifacts: []
    })
    assert.throws(() => decodeBundle(artifactsOnly, {
      allowLegacyManifest: false
    }), /manifest.requires is required/)
  })

  it('keeps an immutable pre-canonical published artifact readable', () => {
    const decoded = validateBundle(HISTORICAL_PRE_CANONICAL_BUNDLE)
    assert.equal(decoded.manifest.typeId, 'wlearn.liblinear.classifier@1')
    assert.throws(() => validateBundle(HISTORICAL_PRE_CANONICAL_BUNDLE, {
      allowLegacyManifest: false
    }), /must contain exactly|manifest.artifacts is required/)
  })

  it('rejects extensions to fixed records only in strict canonical mode', () => {
    const hash = sha256Sync(new Uint8Array(0))
    const entry = {
      id: 'a', offset: 0, length: 0, sha256: hash,
      mediaType: 'application/octet-stream'
    }
    const declaration = {
      id: 'a', length: 0, sha256: hash,
      mediaType: 'application/octet-stream'
    }
    const manifest = {
      typeId: 'wlearn.fixed-records@1', bundleVersion: 1,
      requires: [], params: {}, artifacts: [declaration]
    }

    const tocExtension = makeRawBundle(manifest, [{ ...entry, extension: 'legacy' }])
    assert.doesNotThrow(() => decodeBundle(tocExtension))
    assert.throws(() => decodeBundle(tocExtension, {
      allowLegacyManifest: false
    }), /must contain exactly/)

    const declarationExtension = makeRawBundle({
      ...manifest, artifacts: [{ ...declaration, extension: 'legacy' }]
    }, [entry])
    assert.doesNotThrow(() => decodeBundle(declarationExtension))
    assert.throws(() => decodeBundle(declarationExtension, {
      allowLegacyManifest: false
    }), /must contain exactly/)
  })

  it('validates the portable JSON domain for the entire TOC', () => {
    const hash = sha256Sync(new Uint8Array(0))
    const entry = {
      id: 'a', offset: 0, length: 0, sha256: hash,
      mediaType: 'application/octet-stream',
      extension: Number.MAX_SAFE_INTEGER + 1
    }
    assert.throws(() => decodeBundle(makeRawBundle({
      typeId: 'wlearn.toc-json@1', bundleVersion: 1
    }, [entry])), /portable safe range/)
  })

  it('enforces configurable reader limits before parsing', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    assert.throws(() => decodeBundle(bytes, {
      maxBundleBytes: bytes.length - 1
    }), /Bundle exceeds/)
    assert.throws(() => decodeBundle(bytes, {
      limits: { maxManifestBytes: 1 }
    }), /Manifest exceeds/)
  })

  it('rejects unreferenced gaps and trailing bytes', () => {
    const bytes = encodeBundle({ typeId: 'wlearn.test@1' }, [])
    const withTrailing = new Uint8Array(bytes.length + 1)
    withTrailing.set(bytes)
    withTrailing[withTrailing.length - 1] = 0xaa
    assert.throws(() => validateBundle(withTrailing), /trailing blob bytes/)

    const gap = makeRawBundle(
      { typeId: 'wlearn.test@1', bundleVersion: 1 },
      [{ id: 'empty', offset: 1, length: 0, sha256: 'a'.repeat(64) }],
      new Uint8Array([0])
    )
    assert.throws(() => decodeBundle(gap), /blob gap/)
  })
})
