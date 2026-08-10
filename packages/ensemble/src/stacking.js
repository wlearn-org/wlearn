const {
  encodeBundle, validateBundle, register, load: registryLoad,
  assertRequiredLoaders,
  normalizeX, normalizeY, accuracy, r2Score,
  stratifiedKFold, kFold,
  ValidationError, NotFittedError, DisposedError,
  lift
} = require('@wlearn/core')

const TYPE_ID_CLS = 'wlearn.ensemble.stacking.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.stacking.regressor@1'
let _registered = false

class StackingEnsemble {
  #baseSpecs      // [name, Class, params][]
  #metaSpec       // [name, Class, params]
  #baseModels     // fitted base model instances (on full data)
  #metaModel      // fitted meta-model instance
  #cv
  #task
  #passthrough
  #seed
  #classes
  #nClasses
  #nMetaCols
  #fitted = false
  #disposed = false

  constructor(params) {
    this.#baseSpecs = params.estimators || []
    this.#metaSpec = params.finalEstimator || null
    this.#cv = params.cv || 5
    this.#task = params.task || 'classification'
    this.#passthrough = params.passthrough || false
    this.#seed = params.seed ?? 42
    this.#baseModels = null
    this.#metaModel = null
    this.#classes = null
    this.#nClasses = 0
    this.#nMetaCols = 0
    StackingEnsemble._register()
  }

  static async create(params = {}) {
    return new StackingEnsemble(params)
  }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('StackingEnsemble has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('StackingEnsemble is not fitted. Call fit() first.')
  }

  async fit(X, y) {
    this.#ensureAlive()
    if (!this.#metaSpec) {
      throw new ValidationError('StackingEnsemble requires a finalEstimator')
    }

    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const n = Xn.rows

    // Discover classes without changing the currently fitted state.
    let classes = null
    let nClasses = 0
    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      classes = new Int32Array([...labelSet].sort((a, b) => a - b))
      nClasses = classes.length
    }

    // Generate folds
    const folds = this.#task === 'classification'
      ? stratifiedKFold(yn, this.#cv, { shuffle: true, seed: this.#seed })
      : kFold(n, this.#cv, { shuffle: true, seed: this.#seed })

    // Step 1: Generate OOF predictions for each base model
    const nBase = this.#baseSpecs.length
    const colsPerModel = this.#task === 'classification' ? nClasses : 1
    const oofCols = nBase * colsPerModel
    const oofData = new Float64Array(n * oofCols)

    for (let b = 0; b < nBase; b++) {
      const [, EstClass, params] = this.#baseSpecs[b]
      for (const { train, test } of folds) {
        const Xtrain = _subsetX(Xn, train)
        const ytrain = _subsetY(yn, train)
        const Xtest = _subsetX(Xn, test)

        const model = await EstClass.create(params || {})
        let operationError = null
        try {
          model.fit(Xtrain, ytrain)
          if (this.#task === 'classification') {
            const proba = await model.predictProba(Xtest)
            for (let i = 0; i < test.length; i++) {
              const row = test[i]
              for (let c = 0; c < nClasses; c++) {
                oofData[row * oofCols + b * colsPerModel + c] = proba[i * nClasses + c]
              }
            }
          } else {
            const preds = await model.predict(Xtest)
            for (let i = 0; i < test.length; i++) {
              oofData[test[i] * oofCols + b] = preds[i]
            }
          }
        } catch (error) {
          operationError = error
          throw error
        } finally {
          _disposeOwned([model], operationError)
        }
      }
    }

    // Step 2: Build meta-feature matrix
    let metaX
    let nMetaCols
    if (this.#passthrough) {
      nMetaCols = oofCols + Xn.cols
      const metaData = new Float64Array(n * nMetaCols)
      for (let i = 0; i < n; i++) {
        // OOF predictions
        metaData.set(
          oofData.subarray(i * oofCols, (i + 1) * oofCols),
          i * nMetaCols
        )
        // Original features
        metaData.set(
          Xn.data.subarray(i * Xn.cols, (i + 1) * Xn.cols),
          i * nMetaCols + oofCols
        )
      }
      metaX = { data: metaData, rows: n, cols: nMetaCols }
    } else {
      nMetaCols = oofCols
      metaX = { data: oofData, rows: n, cols: oofCols }
    }

    // Steps 3-4 build replacement state transactionally. Every successful
    // create transfers ownership immediately, before fit can fail.
    const baseModels = []
    let metaModel = null
    try {
      for (const [, EstClass, params] of this.#baseSpecs) {
        const model = await EstClass.create(params || {})
        baseModels.push(model)
        model.fit(Xn, yn)
      }

      const [, MetaClass, metaParams] = this.#metaSpec
      metaModel = await MetaClass.create(metaParams || {})
      metaModel.fit(metaX, yn)
    } catch (error) {
      _disposeOwned([...baseModels, metaModel], error)
      throw error
    }

    const previous = [...(this.#baseModels || []), this.#metaModel]
    this.#baseModels = baseModels
    this.#metaModel = metaModel
    this.#classes = classes
    this.#nClasses = nClasses
    this.#nMetaCols = nMetaCols
    this.#fitted = true
    _disposeOwned(previous)
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const metaX = this.#buildMetaFeatures(X)
    return lift(metaX, mx => this.#metaModel.predict(mx))
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'classification') {
      throw new ValidationError('predictProba is only available for classification')
    }
    if (typeof this.#metaModel.predictProba !== 'function') {
      throw new ValidationError('Meta-model does not support predictProba')
    }
    const metaX = this.#buildMetaFeatures(X)
    return lift(metaX, mx => this.#metaModel.predictProba(mx))
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
        cv: this.#cv,
        passthrough: this.#passthrough,
        seed: this.#seed,
        estimatorNames: this.#baseSpecs.map(s => s[0]),
        metaName: this.#metaSpec[0],
        classes: this.#classes ? [...this.#classes] : null,
        nMetaCols: this.#nMetaCols,
      },
    }
    const artifacts = this.#baseModels.map((model, i) => ({
      id: this.#baseSpecs[i][0],
      data: model.save(),
      mediaType: 'application/x-wlearn-bundle',
    }))
    artifacts.push({
      id: this.#metaSpec[0],
      data: this.#metaModel.save(),
      mediaType: 'application/x-wlearn-bundle',
    })
    return encodeBundle(manifest, artifacts)
  }

  static async load(bytes, options = {}) {
    const { manifest } = validateBundle(bytes)
    if (manifest.typeId !== TYPE_ID_CLS && manifest.typeId !== TYPE_ID_REG) {
      throw new ValidationError(
        `StackingEnsemble.load expected typeId "${TYPE_ID_CLS}" or "${TYPE_ID_REG}", got "${manifest.typeId}"`
      )
    }
    StackingEnsemble._register()
    return registryLoad(bytes, options)
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    _disposeOwned([...(this.#baseModels || []), this.#metaModel])
  }

  getParams() {
    return {
      task: this.#task,
      cv: this.#cv,
      passthrough: this.#passthrough,
      seed: this.#seed,
      estimatorNames: this.#baseSpecs.map(s => s[0]),
      metaName: this.#metaSpec ? this.#metaSpec[0] : null,
    }
  }

  setParams(p) {
    this.#ensureAlive()
    if (p.cv !== undefined) this.#cv = p.cv
    if (p.passthrough !== undefined) this.#passthrough = p.passthrough
    if (p.seed !== undefined) this.#seed = p.seed
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

  get isFitted() { return this.#fitted }
  get classes() { return this.#classes }

  // --- Private helpers ---

  #buildMetaFeatures(X) {
    const Xn = normalizeX(X)
    const n = Xn.rows
    const nBase = this.#baseModels.length
    const colsPerModel = this.#task === 'classification' ? this.#nClasses : 1
    const oofCols = nBase * colsPerModel

    // Collect predictions from all base models
    const rawOutputs = []
    let hasPromise = false
    for (let b = 0; b < nBase; b++) {
      const out = this.#task === 'classification'
        ? this.#baseModels[b].predictProba(Xn)
        : this.#baseModels[b].predict(Xn)
      if (out != null && typeof out.then === 'function') hasPromise = true
      rawOutputs.push(out)
    }

    const assemble = (outputs) => {
      const metaData = new Float64Array(n * this.#nMetaCols)
      for (let b = 0; b < nBase; b++) {
        if (this.#task === 'classification') {
          const proba = outputs[b]
          for (let i = 0; i < n; i++) {
            for (let c = 0; c < this.#nClasses; c++) {
              metaData[i * this.#nMetaCols + b * colsPerModel + c] = proba[i * this.#nClasses + c]
            }
          }
        } else {
          const preds = outputs[b]
          for (let i = 0; i < n; i++) {
            metaData[i * this.#nMetaCols + b] = preds[i]
          }
        }
      }
      if (this.#passthrough) {
        for (let i = 0; i < n; i++) {
          metaData.set(
            Xn.data.subarray(i * Xn.cols, (i + 1) * Xn.cols),
            i * this.#nMetaCols + oofCols
          )
        }
      }
      return { data: metaData, rows: n, cols: this.#nMetaCols }
    }

    return hasPromise ? Promise.all(rawOutputs).then(assemble) : assemble(rawOutputs)
  }

  static _register() {
    if (_registered) return
    _registered = true
    const loader = (manifest, toc, blobs, context) => {
      return StackingEnsemble._loadFromParts(manifest, toc, blobs, context)
    }
    register(TYPE_ID_CLS, loader, { acceptsContext: true, sync: false })
    register(TYPE_ID_REG, loader, { acceptsContext: true, sync: false })
  }

  static async _loadFromParts(manifest, toc, blobs, context) {
    const p = manifest.params
    const expectedTypeId = p?.task === 'regression' ? TYPE_ID_REG : TYPE_ID_CLS
    if (manifest.typeId !== expectedTypeId) {
      throw new ValidationError(
        `StackingEnsemble.load expected typeId "${expectedTypeId}", got "${manifest.typeId}"`
      )
    }
    if (!Array.isArray(p.estimatorNames) || typeof p.metaName !== 'string') {
      throw new ValidationError('StackingEnsemble manifest must declare base and meta estimators')
    }
    assertRequiredLoaders(manifest)
    const ens = new StackingEnsemble({
      task: p.task,
      cv: p.cv,
      passthrough: p.passthrough,
      seed: p.seed,
    })
    ens.#classes = p.classes ? new Int32Array(p.classes) : null
    ens.#nClasses = ens.#classes ? ens.#classes.length : 0
    ens.#nMetaCols = p.nMetaCols
    ens.#baseSpecs = p.estimatorNames.map(name => [name, null, null])
    ens.#metaSpec = [p.metaName, null, null]

    // Load base models
    ens.#baseModels = []
    try {
      for (const name of p.estimatorNames) {
        const entry = toc.find(t => t.id === name)
        if (!entry) throw new ValidationError(`No artifact for base estimator "${name}"`)
        const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
        ens.#baseModels.push(await registryLoad(blob, context))
      }

      // Load meta-model
      const metaEntry = toc.find(t => t.id === p.metaName)
      if (!metaEntry) throw new ValidationError(`No artifact for meta estimator "${p.metaName}"`)
      const metaBlob = blobs.subarray(metaEntry.offset, metaEntry.offset + metaEntry.length)
      ens.#metaModel = await registryLoad(metaBlob, context)

      ens.#fitted = true
      return ens
    } catch (error) {
      _disposeLoaded([...ens.#baseModels, ens.#metaModel])
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

// --- Subset helpers ---

function _subsetX(X, indices) {
  const { data, cols } = X
  const rows = indices.length
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const srcOff = indices[i] * cols
    out.set(data.subarray(srcOff, srcOff + cols), i * cols)
  }
  return { data: out, rows, cols }
}

function _subsetY(y, indices) {
  const out = new (y.constructor)(indices.length)
  for (let i = 0; i < indices.length; i++) {
    out[i] = y[indices[i]]
  }
  return out
}

module.exports = { StackingEnsemble }
