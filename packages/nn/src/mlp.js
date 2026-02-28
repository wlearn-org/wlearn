/**
 * MLPClassifier and MLPRegressor for wlearn using polygrad Instance backend.
 *
 * Uses polygrad's MLP family builder (C) for graph construction, deterministic
 * weight init, forward/backward compilation, and optimizer state. Bundles
 * contain IR + safetensors weights for cross-language portability.
 */

import { createRequire } from 'module'
import {
  encodeBundle, decodeBundle, register,
  DisposedError, NotFittedError
} from '@wlearn/core'

const require = createRequire(import.meta.url)

let _Instance = null
let _OPTIM_SGD = null
let _OPTIM_ADAM = null

function getPolygrad() {
  if (!_Instance) {
    const pg = require('polygrad/src/instance')
    _Instance = pg.Instance
    _OPTIM_SGD = pg.OPTIM_SGD
    _OPTIM_ADAM = pg.OPTIM_ADAM
  }
  return { Instance: _Instance, OPTIM_SGD: _OPTIM_SGD, OPTIM_ADAM: _OPTIM_ADAM }
}

function softmax(logits) {
  const max = Math.max(...logits)
  const exp = logits.map(v => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(v => v / sum)
}

const _UNFITTED = Symbol('unfitted')

// ─── MLPClassifier ─────────────────────────────────────────────────────

export class MLPClassifier {
  #instance = null
  #params = {}
  #nrClass = 0
  #classes = []
  #nFeatures = 0
  #fitted = false
  #disposed = false

  constructor(instanceOrSentinel, params, nrClass, classes, nFeatures) {
    if (instanceOrSentinel === _UNFITTED) {
      this.#instance = null
      this.#params = { ...params }
      this.#fitted = false
    } else {
      this.#instance = instanceOrSentinel
      this.#params = { ...params }
      this.#nrClass = nrClass || 0
      this.#classes = classes ? [...classes] : []
      this.#nFeatures = nFeatures || 0
      this.#fitted = true
    }
  }

  static async create(params = {}) {
    return new MLPClassifier(_UNFITTED, params)
  }

  fit(X, y) {
    if (this.#disposed) throw new DisposedError('MLPClassifier has been disposed.')

    const { Instance, OPTIM_SGD, OPTIM_ADAM } = getPolygrad()

    // Normalize X to flat Float32Array rows
    const { rows, cols, data } = this.#normalizeX(X)
    const nSamples = rows
    const nFeatures = cols
    this.#nFeatures = nFeatures

    // Detect classes
    const yArr = Array.isArray(y) ? y : [...y]
    const unique = [...new Set(yArr)].sort((a, b) => a - b)
    this.#classes = unique.map(Number)
    this.#nrClass = unique.length
    const classMap = new Map(unique.map((c, i) => [Number(c), i]))

    // One-hot encode targets
    const yOnehot = new Float32Array(nSamples * this.#nrClass)
    for (let i = 0; i < nSamples; i++) {
      yOnehot[i * this.#nrClass + classMap.get(Number(yArr[i]))] = 1.0
    }

    // Build MLP spec
    const hidden = this.#params.hidden_sizes || this.#params.hiddenSizes || [64]
    const activation = this.#params.activation || 'relu'
    const seed = this.#params.seed ?? 42
    const lr = this.#params.lr || 0.01
    const epochs = this.#params.epochs || 100
    const optimizer = this.#params.optimizer || 'adam'

    const layers = [nFeatures, ...hidden, this.#nrClass]
    const spec = {
      layers, activation, bias: true,
      loss: 'cross_entropy', batch_size: 1, seed
    }

    // Create instance
    if (this.#instance) {
      this.#instance.free()
    }
    this.#instance = Instance.mlp(spec)

    // Set optimizer
    const optimKind = optimizer === 'adam' ? OPTIM_ADAM : OPTIM_SGD
    this.#instance.setOptimizer(optimKind, lr)

    // Train
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < nSamples; i++) {
        const xi = data.subarray(i * nFeatures, (i + 1) * nFeatures)
        const yi = yOnehot.subarray(i * this.#nrClass, (i + 1) * this.#nrClass)
        this.#instance.trainStep({ x: xi, y: yi })
      }
    }

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const result = new Float64Array(rows)

    for (let i = 0; i < rows; i++) {
      const xi = data.subarray(i * cols, (i + 1) * cols)
      const out = this.#instance.forward({ x: xi })
      const logits = out.output
      let maxIdx = 0
      for (let j = 1; j < logits.length; j++) {
        if (logits[j] > logits[maxIdx]) maxIdx = j
      }
      result[i] = maxIdx < this.#classes.length ? this.#classes[maxIdx] : maxIdx
    }

    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const nc = this.#nrClass
    const result = new Float64Array(rows * nc)

    for (let i = 0; i < rows; i++) {
      const xi = data.subarray(i * cols, (i + 1) * cols)
      const out = this.#instance.forward({ x: xi })
      const probs = softmax([...out.output])
      for (let j = 0; j < nc; j++) {
        result[i * nc + j] = probs[j]
      }
    }

    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = Array.isArray(y) ? y : [...y]
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === Number(yArr[i])) correct++
    }
    return correct / preds.length
  }

  save() {
    this.#ensureFitted()
    const irBytes = this.#instance.exportIR()
    const wBytes = this.#instance.exportWeights()

    return encodeBundle(
      {
        typeId: 'wlearn.nn.mlp.classifier@1',
        params: this.getParams(),
        metadata: {
          nrClass: this.#nrClass,
          classes: this.#classes,
          nFeatures: this.#nFeatures
        }
      },
      [
        { id: 'ir', data: new Uint8Array(irBytes) },
        { id: 'weights', data: new Uint8Array(wBytes) }
      ]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return MLPClassifier._fromBundle(manifest, toc, blobs)
  }

  static _fromBundle(manifest, toc, blobs) {
    const { Instance } = getPolygrad()

    const irEntry = toc.find(e => e.id === 'ir')
    const wEntry = toc.find(e => e.id === 'weights')
    if (!irEntry || !wEntry) throw new Error('Bundle missing "ir" or "weights" artifact')

    const irBytes = blobs.subarray(irEntry.offset, irEntry.offset + irEntry.length)
    const wBytes = blobs.subarray(wEntry.offset, wEntry.offset + wEntry.length)

    const instance = Instance.fromIR(irBytes, wBytes)
    const params = manifest.params || {}
    const meta = manifest.metadata || {}

    return new MLPClassifier(
      instance, params,
      meta.nrClass || 0,
      meta.classes,
      meta.nFeatures || 0
    )
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#instance) {
      this.#instance.free()
      this.#instance = null
    }
    this.#fitted = false
  }

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  get isFitted() {
    return this.#fitted && !this.#disposed
  }

  get classes() {
    return [...this.#classes]
  }

  get nrClass() {
    return this.#nrClass
  }

  get capabilities() {
    return {
      classifier: true,
      regressor: false,
      predictProba: true,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  static defaultSearchSpace() {
    return {
      hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
      activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
      lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
      epochs: { type: 'int_uniform', low: 10, high: 200 },
      optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
    }
  }

  #ensureFitted() {
    if (this.#disposed) throw new DisposedError('MLPClassifier has been disposed.')
    if (!this.#fitted) throw new NotFittedError('MLPClassifier is not fitted.')
  }

  #normalizeX(X) {
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { rows, cols, data }
    }
    if (X && X.data) {
      const data = X.data instanceof Float32Array ? X.data : new Float32Array(X.data)
      return { rows: X.rows, cols: X.cols, data }
    }
    throw new Error('X must be number[][] or { data, rows, cols }')
  }
}

// ─── MLPRegressor ──────────────────────────────────────────────────────

export class MLPRegressor {
  #instance = null
  #params = {}
  #nFeatures = 0
  #fitted = false
  #disposed = false

  constructor(instanceOrSentinel, params, nFeatures) {
    if (instanceOrSentinel === _UNFITTED) {
      this.#instance = null
      this.#params = { ...params }
      this.#fitted = false
    } else {
      this.#instance = instanceOrSentinel
      this.#params = { ...params }
      this.#nFeatures = nFeatures || 0
      this.#fitted = true
    }
  }

  static async create(params = {}) {
    return new MLPRegressor(_UNFITTED, params)
  }

  fit(X, y) {
    if (this.#disposed) throw new DisposedError('MLPRegressor has been disposed.')

    const { Instance, OPTIM_SGD, OPTIM_ADAM } = getPolygrad()

    const { rows, cols, data } = this.#normalizeX(X)
    const nSamples = rows
    const nFeatures = cols
    this.#nFeatures = nFeatures

    // Normalize y to Float32Array
    const yArr = y instanceof Float32Array ? y : new Float32Array(Array.isArray(y) ? y : [...y])
    const nOutputs = yArr.length / nSamples

    // Build MLP spec
    const hidden = this.#params.hidden_sizes || this.#params.hiddenSizes || [64]
    const activation = this.#params.activation || 'relu'
    const seed = this.#params.seed ?? 42
    const lr = this.#params.lr || 0.01
    const epochs = this.#params.epochs || 100
    const optimizer = this.#params.optimizer || 'adam'

    const layers = [nFeatures, ...hidden, nOutputs]
    const spec = {
      layers, activation, bias: true,
      loss: 'mse', batch_size: 1, seed
    }

    if (this.#instance) {
      this.#instance.free()
    }
    this.#instance = Instance.mlp(spec)

    const optimKind = optimizer === 'adam' ? OPTIM_ADAM : OPTIM_SGD
    this.#instance.setOptimizer(optimKind, lr)

    // Train
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < nSamples; i++) {
        const xi = data.subarray(i * nFeatures, (i + 1) * nFeatures)
        const yi = yArr.subarray(i * nOutputs, (i + 1) * nOutputs)
        this.#instance.trainStep({ x: xi, y: yi })
      }
    }

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const results = []

    for (let i = 0; i < rows; i++) {
      const xi = data.subarray(i * cols, (i + 1) * cols)
      const out = this.#instance.forward({ x: xi })
      results.push([...out.output])
    }

    // Flatten if single output
    if (results[0].length === 1) {
      return new Float64Array(results.map(r => r[0]))
    }
    return new Float64Array(results.flat())
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = y instanceof Float64Array ? y : new Float64Array(Array.isArray(y) ? y : [...y])
    const yMean = yArr.reduce((a, b) => a + b, 0) / yArr.length
    let ssRes = 0, ssTot = 0
    for (let i = 0; i < yArr.length; i++) {
      ssRes += (yArr[i] - preds[i]) ** 2
      ssTot += (yArr[i] - yMean) ** 2
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot
  }

  save() {
    this.#ensureFitted()
    const irBytes = this.#instance.exportIR()
    const wBytes = this.#instance.exportWeights()

    return encodeBundle(
      {
        typeId: 'wlearn.nn.mlp.regressor@1',
        params: this.getParams(),
        metadata: {
          nFeatures: this.#nFeatures
        }
      },
      [
        { id: 'ir', data: new Uint8Array(irBytes) },
        { id: 'weights', data: new Uint8Array(wBytes) }
      ]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return MLPRegressor._fromBundle(manifest, toc, blobs)
  }

  static _fromBundle(manifest, toc, blobs) {
    const { Instance } = getPolygrad()

    const irEntry = toc.find(e => e.id === 'ir')
    const wEntry = toc.find(e => e.id === 'weights')
    if (!irEntry || !wEntry) throw new Error('Bundle missing "ir" or "weights" artifact')

    const irBytes = blobs.subarray(irEntry.offset, irEntry.offset + irEntry.length)
    const wBytes = blobs.subarray(wEntry.offset, wEntry.offset + wEntry.length)

    const instance = Instance.fromIR(irBytes, wBytes)
    const params = manifest.params || {}
    const meta = manifest.metadata || {}

    return new MLPRegressor(
      instance, params,
      meta.nFeatures || 0
    )
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#instance) {
      this.#instance.free()
      this.#instance = null
    }
    this.#fitted = false
  }

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  get isFitted() {
    return this.#fitted && !this.#disposed
  }

  get capabilities() {
    return {
      classifier: false,
      regressor: true,
      predictProba: false,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  static defaultSearchSpace() {
    return {
      hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
      activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
      lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
      epochs: { type: 'int_uniform', low: 10, high: 200 },
      optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
    }
  }

  #ensureFitted() {
    if (this.#disposed) throw new DisposedError('MLPRegressor has been disposed.')
    if (!this.#fitted) throw new NotFittedError('MLPRegressor is not fitted.')
  }

  #normalizeX(X) {
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { rows, cols, data }
    }
    if (X && X.data) {
      const data = X.data instanceof Float32Array ? X.data : new Float32Array(X.data)
      return { rows: X.rows, cols: X.cols, data }
    }
    throw new Error('X must be number[][] or { data, rows, cols }')
  }
}

// ─── Register loaders ──────────────────────────────────────────────────

register('wlearn.nn.mlp.classifier@1', (m, t, b) => MLPClassifier._fromBundle(m, t, b))
register('wlearn.nn.mlp.regressor@1', (m, t, b) => MLPRegressor._fromBundle(m, t, b))
