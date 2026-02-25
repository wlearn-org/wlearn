import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { sha256Sync } from '../src/hash.js'

describe('sha256Sync', () => {
  // NIST test vectors
  it('hashes empty input', () => {
    const result = sha256Sync(new Uint8Array(0))
    assert.equal(result, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
  })

  it('hashes "abc"', () => {
    const input = new TextEncoder().encode('abc')
    const result = sha256Sync(input)
    assert.equal(result, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
  })

  it('hashes 448-bit message', () => {
    const input = new TextEncoder().encode('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq')
    const result = sha256Sync(input)
    assert.equal(result, '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1')
  })

  it('hashes 896-bit message', () => {
    const input = new TextEncoder().encode('abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu')
    const result = sha256Sync(input)
    assert.equal(result, 'cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1')
  })

  it('handles non-Uint8Array input', () => {
    const buf = new ArrayBuffer(3)
    new Uint8Array(buf).set([0x61, 0x62, 0x63]) // 'abc'
    const result = sha256Sync(buf)
    assert.equal(result, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
  })

  it('produces deterministic output', () => {
    const input = new TextEncoder().encode('test determinism')
    const a = sha256Sync(input)
    const b = sha256Sync(input)
    assert.equal(a, b)
  })
})
