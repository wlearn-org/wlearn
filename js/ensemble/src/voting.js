const {
  encodeBundle, validateBundle, register, load: registryLoad,
  assertRequiredLoaders,
  normalizeX, normalizeY, accuracy, r2Score,
  ValidationError, NotFittedError, DisposedError,
  lift
} = require('@wlearn/core')

const TYPE_ID_CLS = 'wlearn.ensemble.voting.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.voting.regressor@1'
const { validateVotingManifest } = require('./manifest.js')
const {
  classColumnMap, requireProbabilityModel, validateLabelOutput,
  validateProbabilityOutput, validateRegressionOutput
} = require('./class-order.js')
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
  #fitInProgress = false

  constructor(params = {}) {
    this.#specs = params.estimators ?? []
    this.#weights = params.weights ?? null
    this.#voting = params.voting ?? 'soft'
    this.#task = params.task ?? 'classification'
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
    if (this.#fitInProgress) {
      throw new ValidationError('VotingEnsemble fit is already in progress')
    }
    this.#fitInProgress = true
    try {
      return await this.#fitOnce(X, y)
    } finally {
      this.#fitInProgress = false
    }
  }

  async #fitOnce(X, y) {
    const weights = _validateVotingConfig(
      this.#specs, this.#weights, this.#voting, this.#task
    )
    const Xn = normalizeX(X)
    const yn = normalizeY(y)

    let classes = this.#classes
    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      classes = new Int32Array([...labelSet].sort((a, b) => a - b))
    }

    // Build replacement state transactionally. A model becomes owned as soon
    // as create() succeeds, before fit() can fail.
    const models = []
    try {
      for (const [name, EstClass, params] of this.#specs) {
        const model = await EstClass.create(params || {})
        models.push(model)
        await model.fit(Xn, yn)
        if (this.#task === 'classification' && this.#voting === 'soft') {
          _validateSoftVotingModel(
            model, classes, `VotingEnsemble estimator "${name}"`
          )
        }
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
    _disposeReplaced(previous)
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
        const out = new Int32Array(n)
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
        const label = `VotingEnsemble estimator "${this.#specs[m][0]}"`
        const proba = validateProbabilityOutput(outputs[m], n, nc, label)
        const columns = classColumnMap(this.#models[m], this.#classes, label)
        const w = this.#weights[m]
        for (let row = 0; row < n; row++) {
          for (let column = 0; column < nc; column++) {
            result[row * nc + column] +=
              w * proba[row * nc + columns[column]]
          }
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
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot dispose VotingEnsemble while fit is in progress'
      )
    }
    this.#disposed = true
    try {
      _disposeOwned(this.#models || [])
    } finally {
      this.#models = null
      this.#fitted = false
    }
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
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot set VotingEnsemble params while fit is in progress'
      )
    }
    if (!p || typeof p !== 'object' || Array.isArray(p)) {
      throw new ValidationError('VotingEnsemble params must be an object')
    }
    for (const name of Object.keys(p)) {
      if (name !== 'voting' && name !== 'weights') {
        throw new ValidationError(`Unknown VotingEnsemble parameter "${name}"`)
      }
    }
    const voting = p.voting !== undefined ? p.voting : this.#voting
    const requestedWeights = p.weights !== undefined ? p.weights : this.#weights
    const weights = _validateVotingConfig(
      this.#specs, requestedWeights, voting, this.#task, false
    )
    if (this.#fitted && this.#task === 'classification' && voting === 'soft') {
      for (let index = 0; index < this.#models.length; index++) {
        _validateSoftVotingModel(
          this.#models[index], this.#classes,
          `VotingEnsemble estimator "${this.#specs[index][0]}"`
        )
      }
    }
    this.#voting = voting
    this.#weights = weights
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

  get isFitted() { return this.#fitted && !this.#disposed }
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
        const preds = validateRegressionOutput(
          outputs[m], n,
          `VotingEnsemble estimator "${this.#specs[m][0]}"`
        )
        const w = this.#weights[m]
        for (let i = 0; i < n; i++) result[i] += w * preds[i]
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
      const result = new Int32Array(n)
      const validated = outputs.map((output, index) => validateLabelOutput(
        output, n, this.#classes,
        `VotingEnsemble estimator "${this.#specs[index][0]}"`
      ))
      for (let i = 0; i < n; i++) {
        const votes = new Float64Array(nc)
        for (let m = 0; m < validated.length; m++) {
          const pred = validated[m][i]
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
    const p = validateVotingManifest(manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
    assertRequiredLoaders(manifest)
    const specs = p.estimatorNames.map(name => [name, null, null])
    const weights = _validateVotingConfig(
      specs, p.weights, p.voting, p.task, false
    )
    const ens = new VotingEnsemble({
      task: p.task,
      voting: p.voting,
      weights,
    })
    ens.#classes = p.classes ? new Int32Array(p.classes) : null
    ens.#specs = specs
    ens.#models = []
    try {
      for (const name of p.estimatorNames) {
        const entry = toc.find(t => t.id === name)
        if (!entry) throw new ValidationError(`No artifact for estimator "${name}"`)
        const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
        const model = await registryLoad(blob, context)
        ens.#models.push(model)
        if (ens.#task === 'classification' && ens.#voting === 'soft') {
          _validateSoftVotingModel(
            model, ens.#classes, `VotingEnsemble estimator "${name}"`
          )
        }
      }
      ens.#fitted = true
      return ens
    } catch (error) {
      _disposeLoaded(ens.#models)
      throw error
    }
  }
}

function _validateSoftVotingModel(model, classes, label) {
  requireProbabilityModel(model, label)
  return classColumnMap(model, classes, label)
}

function _validateVotingConfig(specs, weights, voting, task, requireConstructors = true) {
  if (task !== 'classification' && task !== 'regression') {
    throw new ValidationError('VotingEnsemble task must be "classification" or "regression"')
  }
  if (voting !== 'soft' && voting !== 'hard') {
    throw new ValidationError('VotingEnsemble voting must be "soft" or "hard"')
  }
  if (!Array.isArray(specs) || specs.length === 0) {
    throw new ValidationError('VotingEnsemble estimators must be a nonempty array')
  }
  const names = new Set()
  for (let index = 0; index < specs.length; index++) {
    const spec = specs[index]
    if (!Array.isArray(spec) || spec.length < 2 ||
        typeof spec[0] !== 'string' || spec[0].length === 0 ||
        (requireConstructors && typeof spec[1]?.create !== 'function')) {
      throw new ValidationError(`VotingEnsemble estimator ${index} has an invalid specification`)
    }
    if (names.has(spec[0])) {
      throw new ValidationError('VotingEnsemble estimator names must be unique')
    }
    names.add(spec[0])
  }
  if (weights === null) {
    return new Float64Array(specs.length).fill(1 / specs.length)
  }
  if (!Array.isArray(weights) &&
      !(ArrayBuffer.isView(weights) && !(weights instanceof DataView))) {
    throw new ValidationError('VotingEnsemble weights must be an array of finite numbers')
  }
  if (weights.length !== specs.length) {
    throw new ValidationError('VotingEnsemble weights must match the estimator count')
  }
  const resolved = new Float64Array(weights.length)
  let total = 0
  for (let index = 0; index < weights.length; index++) {
    if (typeof weights[index] !== 'number' || !Number.isFinite(weights[index]) ||
        weights[index] < 0) {
      throw new ValidationError(
        'VotingEnsemble weights must contain only nonnegative finite numbers'
      )
    }
    resolved[index] = weights[index]
    total += weights[index]
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new ValidationError('VotingEnsemble weights must have a positive finite sum')
  }
  for (let index = 0; index < resolved.length; index++) resolved[index] /= total
  return resolved
}

function _disposeLoaded(models) {
  _disposeOwned(models, new Error('preserve load error'))
}

function _disposeReplaced(models) {
  _disposeOwned(models, new Error('replacement already committed'))
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
