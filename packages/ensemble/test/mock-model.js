const { encodeBundle, decodeBundle, register, normalizeX, normalizeY } = require('@wlearn/core')

const TYPE_ID = 'wlearn.test.mock@1'
let _registered = false

/**
 * Mock estimator for ensemble tests.
 * Classification: predicts majority class, uniform predictProba.
 * Regression: predicts mean of y.
 * Supports full save/load via WLRN bundles.
 */
class MockModel {
  #params
  #fitted = false
  #disposed = false
  #label = 0
  #classes = null
  #nClasses = 0
  #task

  constructor(params = {}) {
    this.#params = { bias: 0, task: 'classification', ...params }
    this.#task = this.#params.task
    MockModel._register()
  }

  static async create(params = {}) {
    return new MockModel(params)
  }

  fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    if (this.#task === 'classification') {
      const counts = new Map()
      for (let i = 0; i < yn.length; i++) {
        counts.set(yn[i], (counts.get(yn[i]) || 0) + 1)
      }
      let best = yn[0], bestC = 0
      for (const [k, v] of counts) {
        if (v > bestC) { best = k; bestC = v }
      }
      this.#label = best + (this.#params.bias || 0)
      this.#classes = new Int32Array([...counts.keys()].sort((a, b) => a - b))
      this.#nClasses = this.#classes.length
    } else {
      let sum = 0
      for (let i = 0; i < yn.length; i++) sum += yn[i]
      this.#label = sum / yn.length + (this.#params.bias || 0)
    }
    this.#fitted = true
    return this
  }

  predict(X) {
    const Xn = normalizeX(X)
    return new Float64Array(Xn.rows).fill(this.#label)
  }

  predictProba(X) {
    const Xn = normalizeX(X)
    const n = Xn.rows
    const nc = this.#nClasses
    const out = new Float64Array(n * nc)
    // Put most weight on predicted class
    const classIdx = this.#classes.indexOf(this.#label)
    for (let i = 0; i < n; i++) {
      const base = 0.1 / (nc - 1 || 1)
      for (let c = 0; c < nc; c++) {
        out[i * nc + c] = c === classIdx ? 0.9 : base
      }
    }
    return out
  }

  score(X, y) {
    const preds = this.predict(X)
    const yn = normalizeY(y)
    if (this.#task === 'classification') {
      let correct = 0
      for (let i = 0; i < yn.length; i++) {
        if (preds[i] === yn[i]) correct++
      }
      return correct / yn.length
    }
    // r2
    let mean = 0
    for (let i = 0; i < yn.length; i++) mean += yn[i]
    mean /= yn.length
    let ssTot = 0, ssRes = 0
    for (let i = 0; i < yn.length; i++) {
      ssTot += (yn[i] - mean) ** 2
      ssRes += (yn[i] - preds[i]) ** 2
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot
  }

  save() {
    const manifest = {
      typeId: TYPE_ID,
      params: this.getParams(),
    }
    const state = JSON.stringify({
      label: this.#label,
      classes: this.#classes ? [...this.#classes] : null,
      nClasses: this.#nClasses,
      task: this.#task,
    })
    const blob = new TextEncoder().encode(state)
    return encodeBundle(manifest, [{ id: 'state', data: blob }])
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    const entry = toc.find(t => t.id === 'state')
    const stateBytes = blobs.subarray(entry.offset, entry.offset + entry.length)
    const state = JSON.parse(new TextDecoder().decode(stateBytes))
    const m = new MockModel(manifest.params)
    m.#label = state.label
    m.#classes = state.classes ? new Int32Array(state.classes) : null
    m.#nClasses = state.nClasses
    m.#task = state.task
    m.#fitted = true
    return m
  }

  dispose() { this.#disposed = true }

  getParams() { return { ...this.#params } }
  setParams(p) { Object.assign(this.#params, p); return this }

  get capabilities() {
    return {
      classifier: this.#task === 'classification',
      regressor: this.#task === 'regression',
      predictProba: this.#task === 'classification',
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
    }
  }

  get isFitted() { return this.#fitted }
  get classes() { return this.#classes }

  static _register() {
    if (_registered) return
    _registered = true
    register(TYPE_ID, (manifest, toc, blobs) => {
      const entry = toc.find(t => t.id === 'state')
      const stateBytes = blobs.subarray(entry.offset, entry.offset + entry.length)
      const state = JSON.parse(new TextDecoder().decode(stateBytes))
      const m = new MockModel(manifest.params)
      m.#label = state.label
      m.#classes = state.classes ? new Int32Array(state.classes) : null
      m.#nClasses = state.nClasses
      m.#task = state.task
      m.#fitted = true
      return m
    })
  }
}

module.exports = { MockModel }
