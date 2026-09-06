// Numeric preprocessing transformers (v1: DenseMatrix only).
// StandardScaler and MinMaxScaler implement the Transformer interface.

const { NotFittedError, DisposedError, ValidationError } = require('./errors.js')
const { normalizeX } = require('./matrix.js')
const { encodeBundle, encodeJSON, decodeJSON } = require('./bundle.js')
const { register } = require('./registry.js')

const STANDARD_SCALER_TYPE_ID_V1 = 'wlearn.preprocess.standard_scaler@1'
const STANDARD_SCALER_TYPE_ID = 'wlearn.preprocess.standard_scaler@2'
const MINMAX_SCALER_TYPE_ID_V1 = 'wlearn.preprocess.minmax_scaler@1'
const MINMAX_SCALER_TYPE_ID = 'wlearn.preprocess.minmax_scaler@2'

function validatedArtifactVectors(artifact, firstKey, secondKey, label) {
  if (artifact === null || typeof artifact !== 'object' || Array.isArray(artifact)) {
    throw new ValidationError(`${label} artifact must be an object`)
  }
  const first = artifact[firstKey]
  const second = artifact[secondKey]
  if (!Array.isArray(first) || !Array.isArray(second) || first.length === 0 ||
      first.length !== second.length) {
    throw new ValidationError(
      `${label} artifact must contain non-empty, equal-length ${firstKey} and ${secondKey}`
    )
  }
  for (let index = 0; index < first.length; index++) {
    if (typeof first[index] !== 'number' || !Number.isFinite(first[index]) ||
        typeof second[index] !== 'number' || !Number.isFinite(second[index])) {
      throw new ValidationError(`${label} artifact statistics must be finite numbers`)
    }
  }
  return [new Float64Array(first), new Float64Array(second)]
}

// --- StandardScaler ---

class StandardScaler {
  #means = null
  #stds = null
  #fitted = false
  #disposed = false
  #params = {}
  #legacyConstantScale = false

  constructor(params = {}) {
    this.#params = { ...params }
  }

  fit(X) {
    this.#ensureAlive()
    const { rows, cols, data } = normalizeX(X)
    if (rows === 0) throw new ValidationError('Cannot fit on empty data')
    if (cols === 0) throw new ValidationError('Cannot fit data with zero columns')

    const means = new Float64Array(cols)
    const m2 = new Float64Array(cols)

    // Welford's online algorithm
    for (let r = 0; r < rows; r++) {
      const n = r + 1
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c]
        if (!Number.isFinite(val)) {
          throw new ValidationError('StandardScaler fit data must contain only finite numbers')
        }
        const delta = val - means[c]
        means[c] += delta / n
        const delta2 = val - means[c]
        m2[c] += delta * delta2
      }
    }

    const stds = new Float64Array(cols)
    for (let c = 0; c < cols; c++) {
      stds[c] = Math.sqrt(m2[c] / rows)
    }

    this.#means = means
    this.#stds = stds
    this.#legacyConstantScale = false
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
        out[idx] = std > 0
          ? (data[idx] - this.#means[c]) / std
          : this.#legacyConstantScale ? 0 : data[idx] - this.#means[c]
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
      {
        typeId: this.#legacyConstantScale
          ? STANDARD_SCALER_TYPE_ID_V1
          : STANDARD_SCALER_TYPE_ID,
        params: this.getParams()
      },
      [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
    )
  }

  static _fromBundle(manifest, toc, blobs) {
    const entry = toc.find(e => e.id === 'params')
    if (!entry) throw new ValidationError('Bundle missing "params" artifact')
    const artifact = decodeJSON(blobs.subarray(entry.offset, entry.offset + entry.length))
    const [means, stds] = validatedArtifactVectors(
      artifact, 'means', 'stds', 'StandardScaler'
    )
    if (stds.some(std => std < 0)) {
      throw new ValidationError('StandardScaler artifact standard deviations must be non-negative')
    }
    const scaler = new StandardScaler(manifest.params || {})
    scaler.#means = means
    scaler.#stds = stds
    scaler.#legacyConstantScale = manifest.typeId === STANDARD_SCALER_TYPE_ID_V1
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
  get capabilities() { return { transformer: true } }
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
  #legacyConstantScale = false

  constructor(params = {}) {
    this.#params = { ...params }
  }

  fit(X) {
    this.#ensureAlive()
    const { rows, cols, data } = normalizeX(X)
    if (rows === 0) throw new ValidationError('Cannot fit on empty data')
    if (cols === 0) throw new ValidationError('Cannot fit data with zero columns')

    const mins = new Float64Array(cols).fill(Infinity)
    const maxs = new Float64Array(cols).fill(-Infinity)

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c]
        if (!Number.isFinite(val)) {
          throw new ValidationError('MinMaxScaler fit data must contain only finite numbers')
        }
        if (val < mins[c]) mins[c] = val
        if (val > maxs[c]) maxs[c] = val
      }
    }

    this.#mins = mins
    this.#maxs = maxs
    this.#legacyConstantScale = false
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
        out[idx] = range > 0
          ? (data[idx] - this.#mins[c]) / range
          : this.#legacyConstantScale ? 0 : data[idx] - this.#mins[c]
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
      {
        typeId: this.#legacyConstantScale
          ? MINMAX_SCALER_TYPE_ID_V1
          : MINMAX_SCALER_TYPE_ID,
        params: this.getParams()
      },
      [{ id: 'params', data: encodeJSON(artifact), mediaType: 'application/json' }]
    )
  }

  static _fromBundle(manifest, toc, blobs) {
    const entry = toc.find(e => e.id === 'params')
    if (!entry) throw new ValidationError('Bundle missing "params" artifact')
    const artifact = decodeJSON(blobs.subarray(entry.offset, entry.offset + entry.length))
    const [mins, maxs] = validatedArtifactVectors(
      artifact, 'mins', 'maxs', 'MinMaxScaler'
    )
    for (let index = 0; index < mins.length; index++) {
      if (maxs[index] < mins[index]) {
        throw new ValidationError('MinMaxScaler artifact maxima must not be below minima')
      }
    }
    const scaler = new MinMaxScaler(manifest.params || {})
    scaler.#mins = mins
    scaler.#maxs = maxs
    scaler.#legacyConstantScale = manifest.typeId === MINMAX_SCALER_TYPE_ID_V1
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
  get capabilities() { return { transformer: true } }
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
register(STANDARD_SCALER_TYPE_ID_V1, StandardScaler._fromBundle, { sync: true })
register(STANDARD_SCALER_TYPE_ID, StandardScaler._fromBundle, { sync: true })
register(MINMAX_SCALER_TYPE_ID_V1, MinMaxScaler._fromBundle, { sync: true })
register(MINMAX_SCALER_TYPE_ID, MinMaxScaler._fromBundle, { sync: true })

module.exports = { StandardScaler, MinMaxScaler }
