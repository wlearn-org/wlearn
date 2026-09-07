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

const TYPE_ID_CLS = 'wlearn.ensemble.stacking.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.stacking.regressor@1'
const { validateStackingManifest } = require('./manifest.js')
const {
  classColumnMap, requireProbabilityModel, validateLabelOutput,
  validateProbabilityOutput, validateRegressionOutput
} = require('./class-order.js')
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
  #fitInProgress = false

  constructor(params = {}) {
    this.#baseSpecs = params.estimators ?? []
    this.#metaSpec = params.finalEstimator || null
    this.#cv = params.cv ?? 5
    this.#task = params.task ?? 'classification'
    this.#passthrough = params.passthrough ?? false
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
    if (this.#fitInProgress) {
      throw new ValidationError('StackingEnsemble fit is already in progress')
    }
    this.#fitInProgress = true
    try {
      return await this.#fitOnce(X, y)
    } finally {
      this.#fitInProgress = false
    }
  }

  async #fitOnce(X, y) {
    _validateStackingConfig(
      this.#baseSpecs, this.#metaSpec, this.#cv, this.#task,
      this.#passthrough, this.#seed
    )

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
    const folds = resolveCv(this.#cv, yn, { task: this.#task, seed: this.#seed, requireComplete: true })

    const baggedBases = []
    const specBases = []
    for (let index = 0; index < this.#baseSpecs.length; index++) {
      const entry = this.#baseSpecs[index]
      if (!Array.isArray(entry)) {
        throw new ValidationError(`Base estimator ${index} must be an array specification`)
      }
      if (entry.length === 2) {
        const [name, model] = entry
        if (!model?.isFitted || !('oofPredictions' in model)) {
          throw new ValidationError(
            `Base estimator "${name}" is a 2-tuple but not a fitted ` +
            'BaggedEstimator with oofPredictions.'
          )
        }
        baggedBases.push([index, name, model])
      } else if (entry.length >= 3 && typeof entry[1]?.create === 'function') {
        specBases.push([index, entry[0], entry[1], entry[2]])
      } else {
        throw new ValidationError(`Base estimator ${index} has an invalid specification`)
      }
    }

    // Step 1: Generate OOF predictions for each base model
    const nBase = this.#baseSpecs.length
    const colsPerModel = this.#task === 'classification' ? nClasses : 1
    const oofCols = nBase * colsPerModel
    const oofData = new Float64Array(n * oofCols)

    for (const [b, name, model] of baggedBases) {
      const baggedParams = typeof model.getParams === 'function'
        ? model.getParams()
        : null
      if (baggedParams?.task !== this.#task) {
        throw new ValidationError(
          `Pre-fitted BaggedEstimator "${name}" task does not match stacking task`
        )
      }
      const oof = _validatePrefittedOof(
        model.oofPredictions, n * colsPerModel,
        `Pre-fitted BaggedEstimator "${name}"`
      )
      if (this.#task === 'classification') {
        const columns = classColumnMap(
          model, classes, `Pre-fitted BaggedEstimator "${name}"`
        )
        for (let row = 0; row < n; row++) {
          for (let column = 0; column < colsPerModel; column++) {
            oofData[row * oofCols + b * colsPerModel + column] =
              oof[row * colsPerModel + columns[column]]
          }
        }
        continue
      }
      for (let row = 0; row < n; row++) {
        for (let column = 0; column < colsPerModel; column++) {
          oofData[row * oofCols + b * colsPerModel + column] =
            oof[row * colsPerModel + column]
        }
      }
    }

    for (const [b, , EstClass, params] of specBases) {
      for (const { train, test } of folds) {
        const Xtrain = subsetRows(Xn, train)
        const ytrain = subsetLabels(yn, train)
        const Xtest = subsetRows(Xn, test)

        const model = await EstClass.create(taskParams(params, this.#task))
        let operationError = null
        try {
          await model.fit(Xtrain, ytrain)
          validateEstimatorTask(model, this.#task)
          if (this.#task === 'classification') {
            const label = `StackingEnsemble base estimator "${this.#baseSpecs[b][0]}"`
            requireProbabilityModel(model, label)
            const columns = classColumnMap(model, classes, label)
            const proba = validateProbabilityOutput(
              await model.predictProba(Xtest), test.length, nClasses, label
            )
            for (let i = 0; i < test.length; i++) {
              const row = test[i]
              for (let c = 0; c < nClasses; c++) {
                oofData[row * oofCols + b * colsPerModel + c] =
                  proba[i * nClasses + columns[c]]
              }
            }
          } else {
            const preds = validateRegressionOutput(
              await model.predict(Xtest), test.length,
              `StackingEnsemble base estimator "${this.#baseSpecs[b][0]}"`
            )
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
    const baseModels = new Array(nBase)
    for (const [index, , model] of baggedBases) baseModels[index] = model
    const createdModels = []
    let metaModel = null
    try {
      for (const [index, , EstClass, params] of specBases) {
        const model = await EstClass.create(taskParams(params, this.#task))
        createdModels.push(model)
        baseModels[index] = model
        await model.fit(Xn, yn)
        validateEstimatorTask(model, this.#task)
        if (this.#task === 'classification') {
          const label =
            `StackingEnsemble base estimator "${this.#baseSpecs[index][0]}"`
          requireProbabilityModel(model, label)
          classColumnMap(
            model, classes, label
          )
        }
      }

      const [, MetaClass, metaParams] = this.#metaSpec
      metaModel = await MetaClass.create(taskParams(metaParams, this.#task))
      await metaModel.fit(metaX, yn)
      validateEstimatorTask(metaModel, this.#task)
      if (this.#task === 'classification') {
        classColumnMap(
          metaModel, classes,
          `StackingEnsemble meta estimator "${this.#metaSpec[0]}"`
        )
      }
    } catch (error) {
      _disposeOwned([...createdModels, metaModel], error)
      throw error
    }

    const previous = [...(this.#baseModels || []), this.#metaModel]
    this.#baseModels = baseModels
    this.#metaModel = metaModel
    this.#classes = classes
    this.#nClasses = nClasses
    this.#nMetaCols = nMetaCols
    this.#fitted = true
    const retained = new Set([...baseModels, metaModel])
    _disposeReplaced(previous.filter(model => model && !retained.has(model)))
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const metaX = this.#buildMetaFeatures(X)
    return lift(metaX, mx => {
      const output = this.#metaModel.predict(mx)
      if (this.#task !== 'classification') {
        return lift(output, predictions => validateRegressionOutput(
          predictions, mx.rows,
          `StackingEnsemble meta estimator "${this.#metaSpec[0]}"`
        ))
      }
      return lift(output, labels => validateLabelOutput(
        labels, mx.rows, this.#classes,
        `StackingEnsemble meta estimator "${this.#metaSpec[0]}"`
      ))
    })
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'classification') {
      throw new ValidationError('predictProba is only available for classification')
    }
    if (!_supportsPredictProba(this.#metaModel)) {
      throw new ValidationError('Meta-model does not support predictProba')
    }
    const metaX = this.#buildMetaFeatures(X)
    return lift(metaX, mx => {
      const output = this.#metaModel.predictProba(mx)
      return lift(output, proba => {
        const rows = mx.rows
        const label = `StackingEnsemble meta estimator "${this.#metaSpec[0]}"`
        const values = validateProbabilityOutput(
          proba, rows, this.#nClasses, label
        )
        const columns = classColumnMap(this.#metaModel, this.#classes, label)
        const aligned = new Float64Array(values.length)
        for (let row = 0; row < rows; row++) {
          for (let column = 0; column < this.#nClasses; column++) {
            aligned[row * this.#nClasses + column] =
              values[row * this.#nClasses + columns[column]]
          }
        }
        return aligned
      })
    })
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
        cv: serializeCv(this.#cv),
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
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot dispose StackingEnsemble while fit is in progress'
      )
    }
    this.#disposed = true
    try {
      _disposeOwned([...(this.#baseModels || []), this.#metaModel])
    } finally {
      this.#baseModels = null
      this.#metaModel = null
      this.#fitted = false
    }
  }

  getParams() {
    return {
      task: this.#task,
      cv: serializeCv(this.#cv),
      passthrough: this.#passthrough,
      seed: this.#seed,
      estimatorNames: this.#baseSpecs.map(s => s[0]),
      metaName: this.#metaSpec ? this.#metaSpec[0] : null,
    }
  }

  setParams(p) {
    this.#ensureAlive()
    if (this.#fitInProgress) {
      throw new ValidationError(
        'Cannot set StackingEnsemble params while fit is in progress'
      )
    }
    if (!p || typeof p !== 'object' || Array.isArray(p)) {
      throw new ValidationError('StackingEnsemble params must be an object')
    }
    for (const name of Object.keys(p)) {
      if (name !== 'cv' && name !== 'passthrough' && name !== 'seed') {
        throw new ValidationError(`Unknown StackingEnsemble parameter "${name}"`)
      }
    }
    const cv = p.cv !== undefined ? p.cv : this.#cv
    const passthrough = p.passthrough !== undefined
      ? p.passthrough
      : this.#passthrough
    const seed = p.seed !== undefined ? p.seed : this.#seed
    _validateStackingConfig(
      this.#baseSpecs, this.#metaSpec, cv, this.#task,
      passthrough, seed, false
    )
    if (p.cv !== undefined || p.passthrough !== undefined || p.seed !== undefined) {
      this.#fitted = false
    }
    this.#cv = cv
    this.#passthrough = passthrough
    this.#seed = seed
    return this
  }

  get capabilities() {
    return {
      classifier: this.#task === 'classification',
      regressor: this.#task === 'regression',
      predictProba: this.#task === 'classification' &&
        _supportsPredictProba(this.#metaModel),
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
    }
  }

  get isFitted() { return this.#fitted && !this.#disposed }
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
          const label = `StackingEnsemble base estimator "${this.#baseSpecs[b][0]}"`
          const proba = validateProbabilityOutput(
            outputs[b], n, this.#nClasses, label
          )
          const columns = classColumnMap(
            this.#baseModels[b], this.#classes, label
          )
          for (let i = 0; i < n; i++) {
            for (let c = 0; c < this.#nClasses; c++) {
              metaData[i * this.#nMetaCols + b * colsPerModel + c] =
                proba[i * this.#nClasses + columns[c]]
            }
          }
        } else {
          const preds = validateRegressionOutput(
            outputs[b], n,
            `StackingEnsemble base estimator "${this.#baseSpecs[b][0]}"`
          )
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
    const p = validateStackingManifest(manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
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
        const model = await registryLoad(blob, context)
        ens.#baseModels.push(model)
        if (ens.#task === 'classification') {
          const label = `StackingEnsemble base estimator "${name}"`
          requireProbabilityModel(model, label)
          classColumnMap(
            model, ens.#classes, label
          )
        }
      }

      // Load meta-model
      const metaEntry = toc.find(t => t.id === p.metaName)
      if (!metaEntry) throw new ValidationError(`No artifact for meta estimator "${p.metaName}"`)
      const metaBlob = blobs.subarray(metaEntry.offset, metaEntry.offset + metaEntry.length)
      ens.#metaModel = await registryLoad(metaBlob, context)
      if (ens.#task === 'classification') {
        classColumnMap(
          ens.#metaModel, ens.#classes,
          `StackingEnsemble meta estimator "${p.metaName}"`
        )
      }

      ens.#fitted = true
      return ens
    } catch (error) {
      _disposeLoaded([...ens.#baseModels, ens.#metaModel])
      throw error
    }
  }
}

function _validatePrefittedOof(value, expectedLength, label) {
  if ((!Array.isArray(value) &&
       !(ArrayBuffer.isView(value) && !(value instanceof DataView))) ||
      value.length !== expectedLength) {
    throw new ValidationError(
      `${label} OOF shape does not match the stacking data`
    )
  }
  for (let index = 0; index < value.length; index++) {
    if (typeof value[index] !== 'number' || !Number.isFinite(value[index])) {
      throw new ValidationError(`${label} OOF predictions must be finite`)
    }
  }
  return value
}

function _supportsPredictProba(model) {
  return typeof model?.predictProba === 'function' &&
    model?.capabilities?.predictProba === true
}

function _validateStackingConfig(
  baseSpecs, metaSpec, cv, task, passthrough, seed,
  requireConstructors = true
) {
  if (task !== 'classification' && task !== 'regression') {
    throw new ValidationError(
      'StackingEnsemble task must be "classification" or "regression"'
    )
  }
  if (typeof cv === 'number' ? !Number.isSafeInteger(cv) || cv < 2 : !(Array.isArray(cv) || cv?.folds)) {
    throw new ValidationError('StackingEnsemble cv must be a safe integer >= 2')
  }
  if (typeof passthrough !== 'boolean') {
    throw new ValidationError('StackingEnsemble passthrough must be a boolean')
  }
  if (!Number.isSafeInteger(seed)) {
    throw new ValidationError('StackingEnsemble seed must be a safe integer')
  }
  if (!Array.isArray(baseSpecs) || baseSpecs.length === 0) {
    throw new ValidationError('StackingEnsemble estimators must be a nonempty array')
  }

  const names = new Set()
  for (let index = 0; index < baseSpecs.length; index++) {
    const spec = baseSpecs[index]
    if (!Array.isArray(spec) || spec.length < 2 ||
        typeof spec[0] !== 'string' || spec[0].length === 0) {
      throw new ValidationError(
        `StackingEnsemble base estimator ${index} has an invalid specification`
      )
    }
    if (requireConstructors) {
      const fittedBag = spec.length === 2 && spec[1]?.isFitted &&
        'oofPredictions' in spec[1]
      if (!fittedBag && (spec.length < 3 || typeof spec[1]?.create !== 'function')) {
        throw new ValidationError(
          `StackingEnsemble base estimator ${index} has an invalid specification`
        )
      }
    }
    if (names.has(spec[0])) {
      throw new ValidationError('StackingEnsemble estimator names must be unique')
    }
    names.add(spec[0])
  }

  if (!Array.isArray(metaSpec) || metaSpec.length < 2 ||
      typeof metaSpec[0] !== 'string' || metaSpec[0].length === 0 ||
      (requireConstructors && typeof metaSpec[1]?.create !== 'function')) {
    throw new ValidationError('StackingEnsemble requires a valid finalEstimator')
  }
  if (names.has(metaSpec[0])) {
    throw new ValidationError(
      'StackingEnsemble finalEstimator name must differ from base estimator names'
    )
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

module.exports = { StackingEnsemble }
