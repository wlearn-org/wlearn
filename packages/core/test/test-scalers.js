const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { StandardScaler, MinMaxScaler } = require('../src/scalers.js')
const { decodeBundle } = require('../src/bundle.js')
const { NotFittedError, DisposedError } = require('../src/errors.js')

function assertClose(a, b, tol = 1e-10) {
  assert.ok(Math.abs(a - b) < tol, `${a} not close to ${b} (tol=${tol})`)
}

describe('StandardScaler', () => {
  const X = {
    rows: 4,
    cols: 2,
    data: new Float64Array([
      1, 10,
      2, 20,
      3, 30,
      4, 40,
    ])
  }

  it('fit + transform produces zero-mean unit-variance columns', () => {
    const scaler = new StandardScaler()
    scaler.fit(X)
    const result = scaler.transform(X)

    assert.equal(result.rows, 4)
    assert.equal(result.cols, 2)

    // Check mean is ~0 for each column
    for (let c = 0; c < 2; c++) {
      let sum = 0
      for (let r = 0; r < 4; r++) sum += result.data[r * 2 + c]
      assertClose(sum / 4, 0, 1e-10)
    }

    // Check std is ~1 for each column
    for (let c = 0; c < 2; c++) {
      let mean = 0
      for (let r = 0; r < 4; r++) mean += result.data[r * 2 + c]
      mean /= 4
      let m2 = 0
      for (let r = 0; r < 4; r++) {
        const d = result.data[r * 2 + c] - mean
        m2 += d * d
      }
      const std = Math.sqrt(m2 / 3) // ddof=1
      assertClose(std, 1, 1e-10)
    }
  })

  it('fitTransform matches fit + transform', () => {
    const s1 = new StandardScaler()
    s1.fit(X)
    const r1 = s1.transform(X)

    const s2 = new StandardScaler()
    const r2 = s2.fitTransform(X)

    assert.deepEqual(r1.data, r2.data)
  })

  it('save + load round-trips', () => {
    const scaler = new StandardScaler()
    scaler.fit(X)
    const origResult = scaler.transform(X)

    const bytes = scaler.save()
    const { manifest, toc, blobs } = decodeBundle(bytes)
    assert.equal(manifest.typeId, 'wlearn.preprocess.standard_scaler@1')

    const loaded = StandardScaler._fromBundle(manifest, toc, blobs)
    const loadedResult = loaded.transform(X)

    assert.deepEqual(origResult.data, loadedResult.data)
  })

  it('handles number[][] input', () => {
    const scaler = new StandardScaler()
    scaler.fit([[1, 10], [2, 20], [3, 30], [4, 40]])
    const result = scaler.transform([[1, 10], [2, 20]])
    assert.equal(result.rows, 2)
    assert.equal(result.cols, 2)
  })

  it('handles constant column (std=0)', () => {
    const scaler = new StandardScaler()
    scaler.fit({ rows: 3, cols: 1, data: new Float64Array([5, 5, 5]) })
    const result = scaler.transform({ rows: 2, cols: 1, data: new Float64Array([5, 10]) })
    // Constant column -> output 0
    assert.equal(result.data[0], 0)
    assert.equal(result.data[1], 0)
  })

  it('throws NotFittedError before fit', () => {
    const scaler = new StandardScaler()
    assert.throws(() => scaler.transform(X), NotFittedError)
  })

  it('throws DisposedError after dispose', () => {
    const scaler = new StandardScaler()
    scaler.fit(X)
    scaler.dispose()
    assert.throws(() => scaler.transform(X), DisposedError)
    assert.equal(scaler.isFitted, false)
  })

  it('rejects column mismatch', () => {
    const scaler = new StandardScaler()
    scaler.fit(X)
    assert.throws(() => scaler.transform({ rows: 1, cols: 3, data: new Float64Array(3) }))
  })

  it('getParams and setParams work', () => {
    const scaler = new StandardScaler({ withMean: true })
    assert.deepEqual(scaler.getParams(), { withMean: true })
    scaler.setParams({ withMean: false })
    assert.deepEqual(scaler.getParams(), { withMean: false })
  })
})

describe('MinMaxScaler', () => {
  const X = {
    rows: 4,
    cols: 2,
    data: new Float64Array([
      1, 10,
      2, 20,
      3, 30,
      4, 40,
    ])
  }

  it('fit + transform scales to [0, 1]', () => {
    const scaler = new MinMaxScaler()
    scaler.fit(X)
    const result = scaler.transform(X)

    assert.equal(result.rows, 4)
    assert.equal(result.cols, 2)

    // First col: 1,2,3,4 -> 0, 1/3, 2/3, 1
    assertClose(result.data[0], 0)
    assertClose(result.data[2], 1 / 3)
    assertClose(result.data[4], 2 / 3)
    assertClose(result.data[6], 1)

    // Second col: 10,20,30,40 -> 0, 1/3, 2/3, 1
    assertClose(result.data[1], 0)
    assertClose(result.data[3], 1 / 3)
    assertClose(result.data[5], 2 / 3)
    assertClose(result.data[7], 1)
  })

  it('fitTransform matches fit + transform', () => {
    const s1 = new MinMaxScaler()
    s1.fit(X)
    const r1 = s1.transform(X)

    const s2 = new MinMaxScaler()
    const r2 = s2.fitTransform(X)

    assert.deepEqual(r1.data, r2.data)
  })

  it('save + load round-trips', () => {
    const scaler = new MinMaxScaler()
    scaler.fit(X)
    const origResult = scaler.transform(X)

    const bytes = scaler.save()
    const { manifest, toc, blobs } = decodeBundle(bytes)
    assert.equal(manifest.typeId, 'wlearn.preprocess.minmax_scaler@1')

    const loaded = MinMaxScaler._fromBundle(manifest, toc, blobs)
    const loadedResult = loaded.transform(X)

    assert.deepEqual(origResult.data, loadedResult.data)
  })

  it('handles constant column (range=0)', () => {
    const scaler = new MinMaxScaler()
    scaler.fit({ rows: 3, cols: 1, data: new Float64Array([5, 5, 5]) })
    const result = scaler.transform({ rows: 2, cols: 1, data: new Float64Array([5, 10]) })
    assert.equal(result.data[0], 0)
    assert.equal(result.data[1], 0)
  })

  it('can scale outside [0,1] for unseen data', () => {
    const scaler = new MinMaxScaler()
    scaler.fit({ rows: 2, cols: 1, data: new Float64Array([0, 10]) })
    const result = scaler.transform({ rows: 1, cols: 1, data: new Float64Array([20]) })
    assertClose(result.data[0], 2) // (20 - 0) / 10 = 2
  })

  it('throws NotFittedError before fit', () => {
    assert.throws(() => new MinMaxScaler().transform(X), NotFittedError)
  })

  it('throws DisposedError after dispose', () => {
    const scaler = new MinMaxScaler()
    scaler.fit(X)
    scaler.dispose()
    assert.throws(() => scaler.transform(X), DisposedError)
  })
})
