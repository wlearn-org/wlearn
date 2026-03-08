const { normalizeX, normalizeY } = require('./matrix.js')
const { ValidationError } = require('./errors.js')

/**
 * ML preprocessing: imputation, encoding, scaling.
 * Learns parameters during fit(), applies during transform().
 * State is serializable for .wlrn bundles.
 *
 * Designed to be used as a Pipeline step before model fitting.
 * Future: delegate to tranfi transforms once fitted state serialization lands.
 */
class Preprocessor {
  #config
  #fitted = false
  // Learned state
  #colTypes = null     // 'numeric' | 'categorical' per column
  #imputeValues = null // per-column impute value (mean for numeric, mode for categorical)
  #encodings = null    // per-column: { type: 'onehot', mapping: { val: colIdx } }
  #scaleParams = null  // per-column: { mean, std } or { min, range }
  #outputCols = 0      // total output columns after encoding

  constructor(config = {}) {
    this.#config = {
      impute: config.impute ?? 'auto',          // 'auto' | 'mean' | 'median' | 'zero' | false
      encode: config.encode ?? 'auto',          // 'auto' | 'onehot' | 'label' | false
      scale: config.scale ?? false,             // 'standard' | 'minmax' | false
      maxCategories: config.maxCategories ?? 20, // columns with more unique values treated as numeric
      ...config,
    }
  }

  /**
   * Learn preprocessing parameters from training data.
   * X is a DenseMatrix { data, rows, cols }.
   */
  fit(X, y) {
    const Xn = normalizeX(X)
    const { data, rows, cols } = Xn

    // Step 1: Detect column types and compute statistics
    this.#colTypes = new Array(cols)
    this.#imputeValues = new Float64Array(cols)
    const colStats = new Array(cols)

    for (let c = 0; c < cols; c++) {
      const values = []
      let hasNaN = false
      let sum = 0
      let count = 0

      for (let r = 0; r < rows; r++) {
        const v = data[r * cols + c]
        if (Number.isNaN(v)) {
          hasNaN = true
        } else {
          values.push(v)
          sum += v
          count++
        }
      }

      // Detect categorical: integer-valued with few unique values
      const unique = new Set(values)
      const isInteger = values.every(v => v === Math.floor(v))
      const isCategorical = this.#config.encode !== false &&
        isInteger && unique.size <= this.#config.maxCategories && unique.size >= 2

      this.#colTypes[c] = isCategorical ? 'categorical' : 'numeric'

      // Compute impute value
      if (this.#config.impute !== false && hasNaN) {
        if (isCategorical) {
          // Mode
          const freq = new Map()
          for (const v of values) freq.set(v, (freq.get(v) || 0) + 1)
          let modeVal = values[0], modeCount = 0
          for (const [v, ct] of freq) {
            if (ct > modeCount) { modeVal = v; modeCount = ct }
          }
          this.#imputeValues[c] = modeVal
        } else {
          // Mean
          this.#imputeValues[c] = count > 0 ? sum / count : 0
        }
      }

      colStats[c] = { values, unique, sum, count }
    }

    // Step 2: Build encodings for categorical columns
    this.#encodings = new Array(cols).fill(null)
    let outputIdx = 0

    if (this.#config.encode !== false) {
      for (let c = 0; c < cols; c++) {
        if (this.#colTypes[c] === 'categorical') {
          const sorted = [...colStats[c].unique].sort((a, b) => a - b)
          if (this.#config.encode === 'label' || this.#config.encode === false) {
            // Label encoding: 1 output column
            const mapping = new Map()
            sorted.forEach((v, i) => mapping.set(v, i))
            this.#encodings[c] = { type: 'label', mapping, startIdx: outputIdx }
            outputIdx++
          } else {
            // One-hot encoding (default for 'auto' and 'onehot')
            const mapping = new Map()
            sorted.forEach((v, i) => mapping.set(v, outputIdx + i))
            this.#encodings[c] = { type: 'onehot', mapping, startIdx: outputIdx, size: sorted.length }
            outputIdx += sorted.length
          }
        } else {
          outputIdx++
        }
      }
    } else {
      outputIdx = cols
    }
    this.#outputCols = outputIdx

    // Step 3: Compute scaling parameters (on non-NaN values)
    this.#scaleParams = new Array(this.#outputCols).fill(null)

    if (this.#config.scale) {
      // Recompute on the output space after encoding
      // For now, scale numeric columns only
      let outC = 0
      for (let c = 0; c < cols; c++) {
        if (this.#colTypes[c] === 'categorical' && this.#config.encode !== false) {
          const enc = this.#encodings[c]
          outC += enc.type === 'onehot' ? enc.size : 1
          continue
        }
        const vals = colStats[c].values
        if (this.#config.scale === 'standard') {
          const mean = colStats[c].sum / (colStats[c].count || 1)
          let variance = 0
          for (const v of vals) variance += (v - mean) ** 2
          variance /= (vals.length || 1)
          const std = Math.sqrt(variance)
          this.#scaleParams[outC] = { mean, std: std || 1 }
        } else if (this.#config.scale === 'minmax') {
          let min = Infinity, max = -Infinity
          for (const v of vals) { if (v < min) min = v; if (v > max) max = v }
          const range = max - min || 1
          this.#scaleParams[outC] = { min, range }
        }
        outC++
      }
    }

    this.#fitted = true
    return this
  }

  /**
   * Apply learned preprocessing to new data.
   * Returns a new DenseMatrix.
   */
  transform(X) {
    if (!this.#fitted) throw new ValidationError('Preprocessor not fitted')
    const Xn = normalizeX(X)
    const { data, rows, cols } = Xn
    const outCols = this.#outputCols
    const out = new Float64Array(rows * outCols)

    for (let r = 0; r < rows; r++) {
      let outC = 0
      for (let c = 0; c < cols; c++) {
        let v = data[r * cols + c]

        // Impute missing
        if (Number.isNaN(v) && this.#config.impute !== false) {
          v = this.#imputeValues[c]
        }

        // Encode
        if (this.#colTypes[c] === 'categorical' && this.#encodings[c]) {
          const enc = this.#encodings[c]
          if (enc.type === 'onehot') {
            // Zero-fill the one-hot range
            const idx = enc.mapping.get(v)
            if (idx !== undefined) {
              out[r * outCols + idx] = 1
            }
            outC = enc.startIdx + enc.size
          } else {
            // Label encoding
            const label = enc.mapping.get(v)
            out[r * outCols + outC] = label !== undefined ? label : -1
            outC++
          }
        } else {
          out[r * outCols + outC] = v
          outC++
        }
      }

      // Apply scaling
      if (this.#config.scale) {
        for (let j = 0; j < outCols; j++) {
          const sp = this.#scaleParams[j]
          if (!sp) continue
          if (this.#config.scale === 'standard') {
            out[r * outCols + j] = (out[r * outCols + j] - sp.mean) / sp.std
          } else if (this.#config.scale === 'minmax') {
            out[r * outCols + j] = (out[r * outCols + j] - sp.min) / sp.range
          }
        }
      }
    }

    return { data: out, rows, cols: outCols }
  }

  /**
   * Fit and transform in one call.
   */
  fitTransform(X, y) {
    this.fit(X, y)
    return this.transform(X)
  }

  /**
   * Serialize fitted state for .wlrn bundle.
   */
  getState() {
    if (!this.#fitted) throw new ValidationError('Preprocessor not fitted')
    return {
      config: this.#config,
      colTypes: this.#colTypes,
      imputeValues: [...this.#imputeValues],
      encodings: this.#encodings.map(e => {
        if (!e) return null
        return {
          type: e.type,
          mapping: [...e.mapping.entries()],
          startIdx: e.startIdx,
          size: e.size,
        }
      }),
      scaleParams: this.#scaleParams,
      outputCols: this.#outputCols,
    }
  }

  /**
   * Restore from serialized state.
   */
  static fromState(state) {
    const pp = new Preprocessor(state.config)
    pp.#colTypes = state.colTypes
    pp.#imputeValues = new Float64Array(state.imputeValues)
    pp.#encodings = state.encodings.map(e => {
      if (!e) return null
      return {
        type: e.type,
        mapping: new Map(e.mapping),
        startIdx: e.startIdx,
        size: e.size,
      }
    })
    pp.#scaleParams = state.scaleParams
    pp.#outputCols = state.outputCols
    pp.#fitted = true
    return pp
  }

  get isFitted() { return this.#fitted }
  get outputCols() { return this.#outputCols }

  get capabilities() {
    return { transformer: true }
  }

  dispose() {}
}

module.exports = { Preprocessor }
