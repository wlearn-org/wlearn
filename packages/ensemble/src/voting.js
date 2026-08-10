const {
  encodeBundle, validateBundle, register, load: registryLoad,
  assertRequiredLoaders,
  normalizeX, normalizeY, accuracy, r2Score,
  ValidationError, NotFittedError, DisposedError,
  lift
} = require('@wlearn/core')

const TYPE_ID_CLS = 'wlearn.ensemble.voting.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.voting.regressor@1'
let _registered = false

class VotingEnsemble {
  #specs       // [name, Class, params][]
  #models      // fitted instances
  #weights
  #voting      // 'soft' | 'hard'
  #task        // 'classification' | 'regression'
  #classes
  #fitted = false
  #disposed = false

  constructor(params) {
    this.#specs = params.estimators || []
    this.#weights = params.weights || null
    this.#voting = params.voting || 'soft'
    this.#task = params.task || 'classification'
    this.#models = null
    this.#classes = null
    VotingEnsemble._register()
  }

  static async create(params = {}) {
    return new VotingEnsemble(params)
  }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('VotingEnsemble has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('VotingEnsemble is not fitted. Call fit() first.')
  }

  async fit(X, y) {
    this.#ensureAlive()
    const Xn = normalizeX(X)
    const yn = normalizeY(y)

    let classes = this.#classes
    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      classes = new Int32Array([...labelSet].sort((a, b) => a - b))
    }

    // Default equal weights
    const weights = this.#weights ||
      new Float64Array(this.#specs.length).fill(1 / this.#specs.length)

    // Build replacement state transactionally. A model becomes owned as soon
    // as create() succeeds, before fit() can fail.
    const models = []
    try {
      for (const [, EstClass, params] of this.#specs) {
        const model = await EstClass.create(params || {})
        models.push(model)
        model.fit(Xn, yn)
      }
    } catch (error) {
      _disposeOwned(models, error)
      throw error
    }

    const previous = this.#models || []
    this.#models = models
    this.#classes = classes
    this.#weights = weights
    this.#fitted = true
    _disposeOwned(previous)
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const Xn = normalizeX(X)
    const n = Xn.rows

    if (this.#task === 'regression') {
      return this.#weightedAverage(Xn, n)
    }

    if (this.#voting === 'soft') {
      const proba = this.predictProba(Xn)
      return lift(proba, p => {
        const nc = this.#classes.length
        const out = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let bestC = 0, bestV = -Infinity
          for (let c = 0; c < nc; c++) {
            if (p[i * nc + c] > bestV) { bestV = p[i * nc + c]; bestC = c }
          }
          out[i] = this.#classes[bestC]
        }
        return out
      })
    }

    // Hard voting: majority vote
    return this.#majorityVote(Xn, n)
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'classification') {
      throw new ValidationError('predictProba is only available for classification')
    }
    if (this.#voting === 'hard') {
      throw new ValidationError('predictProba requires voting="soft"')
    }

    const Xn = normalizeX(X)
    const n = Xn.rows
    const nc = this.#classes.length

    // Collect predictions from all models
    const rawOutputs = []
    let hasPromise = false
    for (let m = 0; m < this.#models.length; m++) {
      const out = this.#models[m].predictProba(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }

    const assemble = (outputs) => {
      const result = new Float64Array(n * nc)
      for (let m = 0; m < outputs.length; m++) {
        const proba = outputs[m]
        const w = this.#weights[m]
        for (let i = 0; i < n * nc; i++) {
          result[i] += w * proba[i]
        }
      }
      return result
    }

    return hasPromise ? Promise.all(rawOutputs).then(assemble) : assemble(rawOutputs)
  }

  score(X, y) {
    this.#ensureFitted()
    const preds = this.predict(X)
    const yn = normalizeY(y)
    const scorer = this.#task === 'classification' ? accuracy : r2Score
    return lift(preds, p => scorer(yn, p))
  }

  save() {
    this.#ensureFitted()
    const typeId = this.#task === 'classification' ? TYPE_ID_CLS : TYPE_ID_REG
    const manifest = {
      typeId,
      params: {
        task: this.#task,
        voting: this.#voting,
        weights: [...this.#weights],
        estimatorNames: this.#specs.map(s => s[0]),
        classes: this.#classes ? [...this.#classes] : null,
      },
    }
    const artifacts = this.#models.map((model, i) => ({
      id: this.#specs[i][0],
      data: model.save(),
      mediaType: 'application/x-wlearn-bundle',
    }))
    return encodeBundle(manifest, artifacts)
  }

  static async load(bytes, options = {}) {
    const { manifest } = validateBundle(bytes)
    if (manifest.typeId !== TYPE_ID_CLS && manifest.typeId !== TYPE_ID_REG) {
      throw new ValidationError(
        `VotingEnsemble.load expected typeId "${TYPE_ID_CLS}" or "${TYPE_ID_REG}", got "${manifest.typeId}"`
      )
    }
    VotingEnsemble._register()
    return registryLoad(bytes, options)
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    _disposeOwned(this.#models || [])
  }

  getParams() {
    return {
      task: this.#task,
      voting: this.#voting,
      weights: this.#weights ? [...this.#weights] : null,
      estimatorNames: this.#specs.map(s => s[0]),
    }
  }

  setParams(p) {
    this.#ensureAlive()
    if (p.voting !== undefined) this.#voting = p.voting
    if (p.weights !== undefined) this.#weights = new Float64Array(p.weights)
    return this
  }

  get capabilities() {
    return {
      classifier: this.#task === 'classification',
      regressor: this.#task === 'regression',
      predictProba: this.#task === 'classification' && this.#voting === 'soft',
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
    }
  }

  get isFitted() { return this.#fitted }
  get classes() { return this.#classes }

  // --- Private helpers ---

  #weightedAverage(Xn, n) {
    const rawOutputs = []
    let hasPromise = false
    for (let m = 0; m < this.#models.length; m++) {
      const out = this.#models[m].predict(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }
    const assemble = (outputs) => {
      const result = new Float64Array(n)
      for (let m = 0; m < outputs.length; m++) {
        const w = this.#weights[m]
        for (let i = 0; i < n; i++) result[i] += w * outputs[m][i]
      }
      return result
    }
    return hasPromise ? Promise.all(rawOutputs).then(assemble) : assemble(rawOutputs)
  }

  #majorityVote(Xn, n) {
    const rawOutputs = []
    let hasPromise = false
    for (let m = 0; m < this.#models.length; m++) {
      const out = this.#models[m].predict(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }
    const assemble = (outputs) => {
      const nc = this.#classes.length
      const result = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const votes = new Float64Array(nc)
        for (let m = 0; m < outputs.length; m++) {
          const pred = outputs[m][i]
          const classIdx = this.#classes.indexOf(pred)
          if (classIdx >= 0) votes[classIdx] += this.#weights[m]
        }
        let bestC = 0, bestV = -Infinity
        for (let c = 0; c < nc; c++) {
          if (votes[c] > bestV) { bestV = votes[c]; bestC = c }
        }
        result[i] = this.#classes[bestC]
      }
      return result
    }
    return hasPromise ? Promise.all(rawOutputs).then(assemble) : assemble(rawOutputs)
  }

  static _register() {
    if (_registered) return
    _registered = true
    const loader = (manifest, toc, blobs, context) => {
      return VotingEnsemble._loadFromParts(manifest, toc, blobs, context)
    }
    register(TYPE_ID_CLS, loader, { acceptsContext: true, sync: false })
    register(TYPE_ID_REG, loader, { acceptsContext: true, sync: false })
  }

  static async _loadFromParts(manifest, toc, blobs, context) {
    const p = manifest.params
    const expectedTypeId = p?.task === 'regression' ? TYPE_ID_REG : TYPE_ID_CLS
    if (manifest.typeId !== expectedTypeId) {
      throw new ValidationError(
        `VotingEnsemble.load expected typeId "${expectedTypeId}", got "${manifest.typeId}"`
      )
    }
    if (!Array.isArray(p.estimatorNames)) {
      throw new ValidationError('VotingEnsemble manifest must declare estimatorNames')
    }
    assertRequiredLoaders(manifest)
    const ens = new VotingEnsemble({
      task: p.task,
      voting: p.voting,
      weights: new Float64Array(p.weights),
    })
    ens.#classes = p.classes ? new Int32Array(p.classes) : null
    ens.#specs = p.estimatorNames.map(name => [name, null, null])
    ens.#models = []
    try {
      for (const name of p.estimatorNames) {
        const entry = toc.find(t => t.id === name)
        if (!entry) throw new ValidationError(`No artifact for estimator "${name}"`)
        const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
        const model = await registryLoad(blob, context)
        ens.#models.push(model)
      }
      ens.#fitted = true
      return ens
    } catch (error) {
      _disposeLoaded(ens.#models)
      throw error
    }
  }
}

function _disposeLoaded(models) {
  _disposeOwned(models, new Error('preserve load error'))
}

function _disposeOwned(models, operationError = null) {
  let firstError = null
  for (let i = models.length - 1; i >= 0; i--) {
    try {
      if (typeof models[i]?.dispose === 'function') models[i].dispose()
    } catch (error) {
      if (firstError === null) firstError = error
    }
  }
  if (operationError === null && firstError !== null) throw firstError
}

module.exports = { VotingEnsemble }
