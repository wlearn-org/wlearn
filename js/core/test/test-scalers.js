const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { StandardScaler, MinMaxScaler } = require('../src/scalers.js')
const { decodeBundle, encodeBundle, encodeJSON } = require('../src/bundle.js')
const { loadSync } = require('../src/registry.js')
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

    // Check population std is ~1 for each column (sklearn-compatible)
    for (let c = 0; c < 2; c++) {
      let mean = 0
      for (let r = 0; r < 4; r++) mean += result.data[r * 2 + c]
      mean /= 4
      let m2 = 0
      for (let r = 0; r < 4; r++) {
        const d = result.data[r * 2 + c] - mean
        m2 += d * d
      }
      const std = Math.sqrt(m2 / 4) // ddof=0
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
    assert.equal(manifest.typeId, 'wlearn.preprocess.standard_scaler@2')

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

  it('uses unit scale for a constant training column', () => {
    const scaler = new StandardScaler()
    scaler.fit({ rows: 3, cols: 1, data: new Float64Array([5, 5, 5]) })
    const result = scaler.transform({ rows: 2, cols: 1, data: new Float64Array([5, 10]) })
    // Training values map to zero; later values retain their displacement.
    assert.equal(result.data[0], 0)
    assert.equal(result.data[1], 5)
  })

  it('preserves legacy @1 constant-column inference until refit', () => {
    const bytes = encodeBundle(
      { typeId: 'wlearn.preprocess.standard_scaler@1', params: {} },
      [{
        id: 'params',
        data: encodeJSON({ means: [5], stds: [0] }),
        mediaType: 'application/json'
      }]
    )
    const scaler = loadSync(bytes)
    const legacy = scaler.transform([[5], [10]])
    assert.deepEqual(Array.from(legacy.data), [0, 0])
    assert.equal(decodeBundle(scaler.save()).manifest.typeId, 'wlearn.preprocess.standard_scaler@1')

    scaler.fit([[5], [5]])
    assert.deepEqual(Array.from(scaler.transform([[5], [10]]).data), [0, 5])
    assert.equal(decodeBundle(scaler.save()).manifest.typeId, 'wlearn.preprocess.standard_scaler@2')
  })

  it('rejects malformed @1 and @2 artifacts', () => {
    const malformed = [
      null,
      { means: [], stds: [] },
      { means: [0, 1], stds: [1] },
      { means: [null], stds: [1] },
      { means: [0], stds: [-1] },
      { stds: [1] },
    ]
    for (const version of [1, 2]) {
      for (const artifact of malformed) {
        const bytes = encodeBundle(
          { typeId: `wlearn.preprocess.standard_scaler@${version}`, params: {} },
          [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
        )
        assert.throws(() => loadSync(bytes), /StandardScaler artifact/)
      }
    }
  })

  it('rejects non-finite fit data without replacing a fitted model', () => {
    const scaler = new StandardScaler().fit([[1], [3]])
    assert.throws(() => scaler.fit([[1], [NaN]]), /finite numbers/)
    assert.throws(
      () => scaler.fit({ rows: 1, cols: 0, data: new Float64Array(0) }),
      /zero columns/
    )
    assert.equal(scaler.transform([[3]]).data[0], 1)
    assert.equal(loadSync(scaler.save()).transform([[3]]).data[0], 1)
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
    assert.equal(manifest.typeId, 'wlearn.preprocess.minmax_scaler@2')

    const loaded = MinMaxScaler._fromBundle(manifest, toc, blobs)
    const loadedResult = loaded.transform(X)

    assert.deepEqual(origResult.data, loadedResult.data)
  })

  it('uses unit scale for a constant training column', () => {
    const scaler = new MinMaxScaler()
    scaler.fit({ rows: 3, cols: 1, data: new Float64Array([5, 5, 5]) })
    const result = scaler.transform({ rows: 2, cols: 1, data: new Float64Array([5, 10]) })
    assert.equal(result.data[0], 0)
    assert.equal(result.data[1], 5)
  })

  it('preserves legacy @1 constant-column inference until refit', () => {
    const bytes = encodeBundle(
      { typeId: 'wlearn.preprocess.minmax_scaler@1', params: {} },
      [{
        id: 'params',
        data: encodeJSON({ mins: [5], maxs: [5] }),
        mediaType: 'application/json'
      }]
    )
    const scaler = loadSync(bytes)
    assert.deepEqual(Array.from(scaler.transform([[5], [10]]).data), [0, 0])
    assert.equal(decodeBundle(scaler.save()).manifest.typeId, 'wlearn.preprocess.minmax_scaler@1')

    scaler.fit([[5], [5]])
    assert.deepEqual(Array.from(scaler.transform([[5], [10]]).data), [0, 5])
    assert.equal(decodeBundle(scaler.save()).manifest.typeId, 'wlearn.preprocess.minmax_scaler@2')
  })

  it('rejects malformed @1 and @2 artifacts', () => {
    const malformed = [
      null,
      { mins: [], maxs: [] },
      { mins: [0, 1], maxs: [1] },
      { mins: [null], maxs: [1] },
      { mins: [2], maxs: [1] },
      { maxs: [1] },
    ]
    for (const version of [1, 2]) {
      for (const artifact of malformed) {
        const bytes = encodeBundle(
          { typeId: `wlearn.preprocess.minmax_scaler@${version}`, params: {} },
          [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
        )
        assert.throws(() => loadSync(bytes), /MinMaxScaler artifact/)
      }
    }
  })

  it('rejects non-finite fit data without replacing a fitted model', () => {
    const scaler = new MinMaxScaler().fit([[1], [3]])
    assert.throws(() => scaler.fit([[1], [Infinity]]), /finite numbers/)
    assert.throws(
      () => scaler.fit({ rows: 1, cols: 0, data: new Float64Array(0) }),
      /zero columns/
    )
    assert.equal(scaler.transform([[3]]).data[0], 1)
    assert.equal(loadSync(scaler.save()).transform([[3]]).data[0], 1)
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
