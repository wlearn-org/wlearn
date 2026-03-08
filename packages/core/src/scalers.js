// Numeric preprocessing transformers (v1: DenseMatrix only).
// StandardScaler and MinMaxScaler implement the Transformer interface.

const { NotFittedError, DisposedError, ValidationError } = require('./errors.js')
const { normalizeX } = require('./matrix.js')
const { encodeBundle, encodeJSON, decodeJSON } = require('./bundle.js')
const { register } = require('./registry.js')

const STANDARD_SCALER_TYPE_ID = 'wlearn.preprocess.standard_scaler@1'
const MINMAX_SCALER_TYPE_ID = 'wlearn.preprocess.minmax_scaler@1'

// --- StandardScaler ---

class StandardScaler {
  #means = null
  #stds = null
  #fitted = false
  #disposed = false
  #params = {}

  constructor(params = {}) {
    this.#params = { ...params }
  }

  fit(X) {
    this.#ensureAlive()
    const { rows, cols, data } = normalizeX(X)
    if (rows === 0) throw new ValidationError('Cannot fit on empty data')

    const means = new Float64Array(cols)
    const m2 = new Float64Array(cols)

    // Welford's online algorithm
    for (let r = 0; r < rows; r++) {
      const n = r + 1
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c]
        const delta = val - means[c]
        means[c] += delta / n
        const delta2 = val - means[c]
        m2[c] += delta * delta2
      }
    }

    const stds = new Float64Array(cols)
    for (let c = 0; c < cols; c++) {
      stds[c] = rows > 1 ? Math.sqrt(m2[c] / (rows - 1)) : 0
    }

    this.#means = means
    this.#stds = stds
    this.#fitted = true
    return this
  }

  transform(X) {
    this.#ensureFitted()
    const { rows, cols, data } = normalizeX(X)
    if (cols !== this.#means.length) {
      throw new ValidationError(
        `Expected ${this.#means.length} columns, got ${cols}`
      )
    }

    const out = new Float64Array(rows * cols)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        const std = this.#stds[c]
        out[idx] = std > 0 ? (data[idx] - this.#means[c]) / std : 0
      }
    }
    return { rows, cols, data: out }
  }

  fitTransform(X) {
    this.fit(X)
    return this.transform(X)
  }

  save() {
    this.#ensureFitted()
    const artifact = {
      means: Array.from(this.#means),
      stds: Array.from(this.#stds),
    }
    return encodeBundle(
      { typeId: STANDARD_SCALER_TYPE_ID, params: this.getParams() },
      [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
    )
  }

  static _fromBundle(manifest, toc, blobs) {
    const entry = toc.find(e => e.id === 'params')
    if (!entry) throw new ValidationError('Bundle missing "params" artifact')
    const artifact = decodeJSON(blobs.subarray(entry.offset, entry.offset + entry.length))
    const scaler = new StandardScaler(manifest.params || {})
    scaler.#means = new Float64Array(artifact.means)
    scaler.#stds = new Float64Array(artifact.stds)
    scaler.#fitted = true
    return scaler
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#means = null
    this.#stds = null
    this.#fitted = false
  }

  getParams() { return { ...this.#params } }
  setParams(p) { Object.assign(this.#params, p); return this }
  get isFitted() { return this.#fitted && !this.#disposed }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('StandardScaler has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('StandardScaler is not fitted.')
  }
}

// --- MinMaxScaler ---

class MinMaxScaler {
  #mins = null
  #maxs = null
  #fitted = false
  #disposed = false
  #params = {}

  constructor(params = {}) {
    this.#params = { ...params }
  }

  fit(X) {
    this.#ensureAlive()
    const { rows, cols, data } = normalizeX(X)
    if (rows === 0) throw new ValidationError('Cannot fit on empty data')

    const mins = new Float64Array(cols).fill(Infinity)
    const maxs = new Float64Array(cols).fill(-Infinity)

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c]
        if (val < mins[c]) mins[c] = val
        if (val > maxs[c]) maxs[c] = val
      }
    }

    this.#mins = mins
    this.#maxs = maxs
    this.#fitted = true
    return this
  }

  transform(X) {
    this.#ensureFitted()
    const { rows, cols, data } = normalizeX(X)
    if (cols !== this.#mins.length) {
      throw new ValidationError(
        `Expected ${this.#mins.length} columns, got ${cols}`
      )
    }

    const out = new Float64Array(rows * cols)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        const range = this.#maxs[c] - this.#mins[c]
        out[idx] = range > 0 ? (data[idx] - this.#mins[c]) / range : 0
      }
    }
    return { rows, cols, data: out }
  }

  fitTransform(X) {
    this.fit(X)
    return this.transform(X)
  }

  save() {
    this.#ensureFitted()
    const artifact = {
      mins: Array.from(this.#mins),
      maxs: Array.from(this.#maxs),
    }
    return encodeBundle(
      { typeId: MINMAX_SCALER_TYPE_ID, params: this.getParams() },
      [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
    )
  }

  static _fromBundle(manifest, toc, blobs) {
    const entry = toc.find(e => e.id === 'params')
    if (!entry) throw new ValidationError('Bundle missing "params" artifact')
    const artifact = decodeJSON(blobs.subarray(entry.offset, entry.offset + entry.length))
    const scaler = new MinMaxScaler(manifest.params || {})
    scaler.#mins = new Float64Array(artifact.mins)
    scaler.#maxs = new Float64Array(artifact.maxs)
    scaler.#fitted = true
    return scaler
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#mins = null
    this.#maxs = null
    this.#fitted = false
  }

  getParams() { return { ...this.#params } }
  setParams(p) { Object.assign(this.#params, p); return this }
  get isFitted() { return this.#fitted && !this.#disposed }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('MinMaxScaler has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('MinMaxScaler is not fitted.')
  }
}

// Auto-register loaders
register(STANDARD_SCALER_TYPE_ID, StandardScaler._fromBundle)
register(MINMAX_SCALER_TYPE_ID, MinMaxScaler._fromBundle)

module.exports = { StandardScaler, MinMaxScaler }
