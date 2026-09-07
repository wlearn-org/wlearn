const { subsetRows, subsetLabels } = require('@wlearn/core')
const { taskParams, validateEstimatorTask, resolveCv, serializeCv } = require('@wlearn/core')
const {
  encodeBundle, validateBundle, register, load: registryLoad,
  assertRequiredLoaders,
  normalizeX, normalizeY, accuracy, r2Score,
  stratifiedKFold, kFold,
  ValidationError, NotFittedError, DisposedError,
  lift
} = require('@wlearn/core')

const TYPE_ID_CLS = 'wlearn.ensemble.bagged.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.bagged.regressor@1'
const { validateBaggingManifest } = require('./manifest.js')
const {
  classColumnMap, requireProbabilityModel, validateProbabilityOutput,
  validateRegressionOutput
} = require('./class-order.js')
let _registered = false

/**
 * K-fold bagged estimator with out-of-fold prediction storage.
 *
 * Trains K * nRepeats copies of a base model. Each repeat uses a different
 * seed for fold assignment. OOF predictions are accumulated (sum + count)
 * and averaged, matching AutoGluon's BaggedEnsembleModel pattern.
 */
class BaggedEstimator {
  #spec         // [name, Class, params]
  #kFold
  #nRepeats
  #task
  #seed
  #foldModels   // fitted model instances, length K * nRepeats
  #classes
  #nClasses = 0
  #nSamples = 0
  #oofAccum     // Float64Array: accumulated OOF predictions (sum)
  #oofCounts    // Uint32Array: per-sample prediction count
  #hasOof = false
  #fitted = false
  #disposed = false
  #fitInProgress = false

  constructor(params = {}) {
    this.#spec = params.estimator || null
    this.#kFold = params.kFold ?? 5
    this.#nRepeats = params.nRepeats ?? 1
    this.#task = params.task ?? 'classification'
    this.#seed = params.seed ?? 42
    this.#foldModels = null
    this.#classes = null
    this.#oofAccum = null
    this.#oofCounts = null
    BaggedEstimator._register()
  }

  static async create(params = {}) {
    return new BaggedEstimator(params)
  }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('BaggedEstimator has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('BaggedEstimator is not fitted. Call fit() first.')
  }

  async fit(X, y) {
    this.#ensureAlive()
    if (this.#fitInProgress) {
      throw new ValidationError('BaggedEstimator fit is already in progress')
    }
    this.#fitInProgress = true
    try {
      return await this.#fitOnce(X, y)
    } finally {
      this.#fitInProgress = false
    }
  }

  async #fitOnce(X, y) {
    _validateBaggingConfig(
      this.#spec, this.#kFold, this.#nRepeats, this.#seed, this.#task
    )
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const n = Xn.rows

    let classes = null
    let nClasses = 0
    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      classes = new Int32Array([...labelSet].sort((a, b) => a - b))
      nClasses = classes.length
    }

    const oofAccum = this.#task === 'classification'
      ? new Float64Array(n * nClasses)
      : new Float64Array(n)
    const oofCounts = new Uint32Array(n)

    const [name, EstClass, params] = this.#spec
    const foldModels = []
    try {
      for (let repeat = 0; repeat < this.#nRepeats; repeat++) {
        const repeatSeed = this.#seed + repeat

        const folds = resolveCv(this.#kFold, yn, { task: this.#task, seed: repeatSeed, requireComplete: true })

        for (const { train, test } of folds) {
          const Xtrain = subsetRows(Xn, train)
          const ytrain = subsetLabels(yn, train)
          const Xtest = subsetRows(Xn, test)

          const model = await EstClass.create(taskParams(params, this.#task))
          foldModels.push(model)
          await model.fit(Xtrain, ytrain)
          validateEstimatorTask(model, this.#task)

          if (this.#task === 'classification') {
            const label = `BaggedEstimator child "${name}"`
            requireProbabilityModel(model, label)
            const columns = classColumnMap(model, classes, label)
            const proba = validateProbabilityOutput(
              await model.predictProba(Xtest), test.length, nClasses, label
            )
            for (let i = 0; i < test.length; i++) {
              const row = test[i]
              for (let c = 0; c < nClasses; c++) {
                oofAccum[row * nClasses + c] +=
                  proba[i * nClasses + columns[c]]
              }
            }
          } else {
            const preds = validateRegressionOutput(
              await model.predict(Xtest), test.length,
              `BaggedEstimator child "${name}"`
            )
            for (let i = 0; i < test.length; i++) {
              oofAccum[test[i]] += preds[i]
            }
          }

          for (let i = 0; i < test.length; i++) {
            oofCounts[test[i]] += 1
          }

        }
      }
    } catch (error) {
      _disposeOwned(foldModels, error)
      throw error
    }

    const previous = this.#foldModels || []
    this.#nSamples = n
    this.#classes = classes
    this.#nClasses = nClasses
    this.#oofAccum = oofAccum
    this.#oofCounts = oofCounts
    this.#hasOof = true
    this.#foldModels = foldModels
    this.#fitted = true
    _disposeReplaced(previous)
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const Xn = normalizeX(X)
    const n = Xn.rows

    if (this.#task === 'regression') {
      return this.#averagePredictions(Xn, n)
    }

    // Classification: average probabilities, then argmax
    const proba = this.predictProba(Xn)
    return lift(proba, p => {
      const nc = this.#nClasses
      const out = new Int32Array(n)
      for (let i = 0; i < n; i++) {
        let bestC = 0, bestV = -Infinity
        for (let c = 0; c < nc; c++) {
          if (p[i * nc + c] > bestV) {
            bestV = p[i * nc + c]
            bestC = c
          }
        }
        out[i] = this.#classes[bestC]
      }
      return out
    })
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'classification') {
      throw new ValidationError('predictProba is only available for classification')
    }

    const Xn = normalizeX(X)
    const n = Xn.rows
    const nc = this.#nClasses
    const nModels = this.#foldModels.length

    const rawOutputs = []
    let hasPromise = false
    for (const model of this.#foldModels) {
      const out = model.predictProba(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }

    const assemble = (outputs) => {
      const result = new Float64Array(n * nc)
      for (let modelIndex = 0; modelIndex < outputs.length; modelIndex++) {
        const label = `BaggedEstimator child ${modelIndex}`
        const proba = validateProbabilityOutput(
          outputs[modelIndex], n, nc, label
        )
        const columns = classColumnMap(
          this.#foldModels[modelIndex], this.#classes, label
        )
        for (let row = 0; row < n; row++) {
          for (let column = 0; column < nc; column++) {
            result[row * nc + column] +=
              proba[row * nc + columns[column]]
          }
        }
      }
      for (let i = 0; i < n * nc; i++) result[i] /= nModels
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

  #averagePredictions(Xn, n) {
    const nModels = this.#foldModels.length
    const rawOutputs = []
    let hasPromise = false
    for (const model of this.#foldModels) {
      const out = model.predict(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }
    const assemble = (outputs) => {
      const result = new Float64Array(n)
      for (let modelIndex = 0; modelIndex < outputs.length; modelIndex++) {
        const preds = validateRegressionOutput(
          outputs[modelIndex], n, `BaggedEstimator child ${modelIndex}`
        )
        for (let i = 0; i < n; i++) result[i] += preds[i]
      }
      for (let i = 0; i < n; i++) result[i] /= nModels
      return result
    }
    return hasPromise ? Promise.all(rawOutputs).then(assemble) : assemble(rawOutputs)
  }

  /**
   * Averaged OOF predictions.
   * Classification: flat (n * nClasses) row-major probabilities.
   * Regression: flat (n) predictions.
   */
  get oofPredictions() {
    this.#ensureFitted()
    if (!this.#hasOof) {
      throw new ValidationError(
        'BaggedEstimator artifact does not include stored OOF predictions'
      )
    }
    const counts = new Uint32Array(this.#oofCounts)
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] === 0) counts[i] = 1
    }

    if (this.#task === 'classification') {
      const nc = this.#nClasses
      const oof = new Float64Array(this.#oofAccum)
      for (let i = 0; i < this.#nSamples; i++) {
        const c = counts[i]
        for (let j = 0; j < nc; j++) {
          oof[i * nc + j] /= c
        }
      }
      return oof
    }

    const oof = new Float64Array(this.#oofAccum)
    for (let i = 0; i < this.#nSamples; i++) {
      oof[i] /= counts[i]
    }
    return oof
  }

  save() {
    this.#ensureFitted()
    const typeId = this.#task === 'classification' ? TYPE_ID_CLS : TYPE_ID_REG

    const manifest = {
      typeId,
      params: {
        task: this.#task,
        kFold: serializeCv(this.#kFold),
        nRepeats: this.#nRepeats,
        seed: this.#seed,
        estimatorName: this.#spec[0],
        classes: this.#classes ? [...this.#classes] : null,
        nClasses: this.#nClasses,
        nSamples: this.#nSamples,
      },
    }

    const artifacts = this.#foldModels.map((model, i) => ({
      id: `fold_${i}`,
      data: model.save(),
      mediaType: 'application/x-wlearn-bundle',
    }))

    // Store OOF data as raw float64 LE bytes
    const oof = this.oofPredictions
    const oofBytes = new Uint8Array(oof.buffer, oof.byteOffset, oof.byteLength)
    artifacts.push({
      id: 'oof',
      data: oofBytes,
      mediaType: 'application/octet-stream',
    })

    return encodeBundle(manifest, artifacts)
  }

  static async load(bytes, options = {}) {
    const { manifest } = validateBundle(bytes)
    if (manifest.typeId !== TYPE_ID_CLS && manifest.typeId !== TYPE_ID_REG) {
      throw new ValidationError(
        `BaggedEstimator.load expected typeId "${TYPE_ID_CLS}" or "${TYPE_ID_REG}", got "${manifest.typeId}"`
      )
    }
    BaggedEstimator._register()
    return registryLoad(bytes, options)
  }

  dispose() {
    if (this.#disposed) return
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot dispose BaggedEstimator while fit is in progress'
      )
    }
    this.#disposed = true
    try {
      _disposeOwned(this.#foldModels || [])
    } finally {
      this.#foldModels = null
      this.#oofAccum = null
      this.#oofCounts = null
      this.#hasOof = false
      this.#fitted = false
    }
  }

  getParams() {
    return {
      task: this.#task,
      kFold: serializeCv(this.#kFold),
      nRepeats: this.#nRepeats,
      seed: this.#seed,
      estimatorName: this.#spec ? this.#spec[0] : null,
    }
  }

  setParams(p) {
    this.#ensureAlive()
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot set BaggedEstimator params while fit is in progress'
      )
    }
    if (!p || typeof p !== 'object' || Array.isArray(p)) {
      throw new ValidationError('BaggedEstimator params must be an object')
    }
    for (const name of Object.keys(p)) {
      if (name !== 'kFold' && name !== 'nRepeats' && name !== 'seed') {
        throw new ValidationError(`Unknown BaggedEstimator parameter "${name}"`)
      }
    }
    const kFold = p.kFold !== undefined ? p.kFold : this.#kFold
    const nRepeats = p.nRepeats !== undefined ? p.nRepeats : this.#nRepeats
    const seed = p.seed !== undefined ? p.seed : this.#seed
    _validateBaggingConfig(
      this.#spec, kFold, nRepeats, seed, this.#task, false
    )
    if (p.kFold !== undefined || p.nRepeats !== undefined || p.seed !== undefined) {
      this.#fitted = false
    }
    this.#kFold = kFold
    this.#nRepeats = nRepeats
    this.#seed = seed
    return this
  }

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

  get isFitted() { return this.#fitted && !this.#disposed }
  get classes() { return this.#classes }

  // --- Static internals ---

  static _register() {
    if (_registered) return
    _registered = true
    const loader = (manifest, toc, blobs, context) =>
      BaggedEstimator._loadFromParts(manifest, toc, blobs, context)
    register(TYPE_ID_CLS, loader, { acceptsContext: true, sync: false })
    register(TYPE_ID_REG, loader, { acceptsContext: true, sync: false })
  }

  static async _loadFromParts(manifest, toc, blobs, context) {
    const p = validateBaggingManifest(manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
    assertRequiredLoaders(manifest)
    const bag = new BaggedEstimator({
      task: p.task,
      kFold: p.kFold,
      nRepeats: p.nRepeats,
      seed: p.seed ?? 42,
    })
    bag.#classes = p.classes ? new Int32Array(p.classes) : null
    bag.#nClasses = p.nClasses || 0
    bag.#nSamples = p.nSamples || 0
    bag.#spec = [p.estimatorName || 'base', null, null]

    // Load fold models
    const nFoldModels = (typeof bag.#kFold === 'number' ? bag.#kFold : bag.#kFold.length) * bag.#nRepeats
    bag.#foldModels = []
    try {
      for (let i = 0; i < nFoldModels; i++) {
        const foldId = `fold_${i}`
        const entry = toc.find(t => t.id === foldId)
        if (!entry) throw new ValidationError(`No artifact for "${foldId}"`)
        const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
        const model = await registryLoad(blob, context)
        bag.#foldModels.push(model)
        if (bag.#task === 'classification') {
          const label = `BaggedEstimator child ${i}`
          requireProbabilityModel(model, label)
          classColumnMap(model, bag.#classes, label)
        }
      }

      // Load OOF data
      const oofEntry = toc.find(t => t.id === 'oof')
      if (oofEntry) {
        const oofBlob = blobs.subarray(oofEntry.offset, oofEntry.offset + oofEntry.length)
        const oof = new Float64Array(
          oofBlob.buffer.slice(oofBlob.byteOffset, oofBlob.byteOffset + oofBlob.byteLength)
        )
        bag.#oofAccum = oof
        bag.#oofCounts = new Uint32Array(bag.#nSamples).fill(1)
        bag.#hasOof = true
      } else {
        if (bag.#task === 'classification') {
          bag.#oofAccum = new Float64Array(bag.#nSamples * bag.#nClasses)
        } else {
          bag.#oofAccum = new Float64Array(bag.#nSamples)
        }
        bag.#oofCounts = new Uint32Array(bag.#nSamples)
        bag.#hasOof = false
      }

      bag.#fitted = true
      return bag
    } catch (error) {
      _disposeLoaded(bag.#foldModels)
      throw error
    }
  }
}

function _validateBaggingConfig(
  spec, kFold, nRepeats, seed, task, requireConstructor = true
) {
  if (task !== 'classification' && task !== 'regression') {
    throw new ValidationError(
      'BaggedEstimator task must be "classification" or "regression"'
    )
  }
  if (typeof kFold === 'number' ? !Number.isSafeInteger(kFold) || kFold < 2 : !(Array.isArray(kFold) || kFold?.folds)) {
    throw new ValidationError('BaggedEstimator kFold must be a safe integer >= 2')
  }
  if (!Number.isSafeInteger(nRepeats) || nRepeats < 1) {
    throw new ValidationError('BaggedEstimator nRepeats must be a safe integer >= 1')
  }
  if (!Number.isSafeInteger((typeof kFold === 'number' ? kFold : (kFold.folds || kFold).length) * nRepeats)) {
    throw new ValidationError('BaggedEstimator fold model count exceeds the safe integer range')
  }
  if (!Number.isSafeInteger(seed) || !Number.isSafeInteger(seed + nRepeats - 1)) {
    throw new ValidationError('BaggedEstimator seed range must contain only safe integers')
  }
  if (!Array.isArray(spec) || spec.length < 2 ||
      typeof spec[0] !== 'string' || spec[0].length === 0 ||
      (requireConstructor && typeof spec[1]?.create !== 'function')) {
    throw new ValidationError('BaggedEstimator requires a valid estimator specification')
  }
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

module.exports = { BaggedEstimator }
