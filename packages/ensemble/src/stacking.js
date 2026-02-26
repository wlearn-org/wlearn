import {
  encodeBundle, decodeBundle, register, load as registryLoad,
  normalizeX, normalizeY, accuracy, r2Score,
  stratifiedKFold, kFold,
  ValidationError, NotFittedError, DisposedError
} from '@wlearn/core'

const TYPE_ID_CLS = 'wlearn.ensemble.stacking.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.stacking.regressor@1'
let _registered = false

export class StackingEnsemble {
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

    // Discover classes
    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      this.#classes = new Int32Array([...labelSet].sort((a, b) => a - b))
      this.#nClasses = this.#classes.length
    }

    // Generate folds
    const folds = this.#task === 'classification'
      ? stratifiedKFold(yn, this.#cv, { shuffle: true, seed: this.#seed })
      : kFold(n, this.#cv, { shuffle: true, seed: this.#seed })

    // Step 1: Generate OOF predictions for each base model
    const nBase = this.#baseSpecs.length
    const colsPerModel = this.#task === 'classification' ? this.#nClasses : 1
    const oofCols = nBase * colsPerModel
    const oofData = new Float64Array(n * oofCols)

    for (let b = 0; b < nBase; b++) {
      const [, EstClass, params] = this.#baseSpecs[b]
      for (const { train, test } of folds) {
        const Xtrain = _subsetX(Xn, train)
        const ytrain = _subsetY(yn, train)
        const Xtest = _subsetX(Xn, test)

        const model = await EstClass.create(params || {})
        try {
          model.fit(Xtrain, ytrain)
          if (this.#task === 'classification') {
            const proba = model.predictProba(Xtest)
            for (let i = 0; i < test.length; i++) {
              const row = test[i]
              for (let c = 0; c < this.#nClasses; c++) {
                oofData[row * oofCols + b * colsPerModel + c] = proba[i * this.#nClasses + c]
              }
            }
          } else {
            const preds = model.predict(Xtest)
            for (let i = 0; i < test.length; i++) {
              oofData[test[i] * oofCols + b] = preds[i]
            }
          }
        } finally {
          model.dispose()
        }
      }
    }

    // Step 2: Build meta-feature matrix
    let metaX
    if (this.#passthrough) {
      this.#nMetaCols = oofCols + Xn.cols
      const metaData = new Float64Array(n * this.#nMetaCols)
      for (let i = 0; i < n; i++) {
        // OOF predictions
        metaData.set(
          oofData.subarray(i * oofCols, (i + 1) * oofCols),
          i * this.#nMetaCols
        )
        // Original features
        metaData.set(
          Xn.data.subarray(i * Xn.cols, (i + 1) * Xn.cols),
          i * this.#nMetaCols + oofCols
        )
      }
      metaX = { data: metaData, rows: n, cols: this.#nMetaCols }
    } else {
      this.#nMetaCols = oofCols
      metaX = { data: oofData, rows: n, cols: oofCols }
    }

    // Step 3: Train base models on full data
    this.#baseModels = []
    for (const [, EstClass, params] of this.#baseSpecs) {
      const model = await EstClass.create(params || {})
      model.fit(Xn, yn)
      this.#baseModels.push(model)
    }

    // Step 4: Train meta-model on OOF features
    const [, MetaClass, metaParams] = this.#metaSpec
    this.#metaModel = await MetaClass.create(metaParams || {})
    this.#metaModel.fit(metaX, yn)

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const metaX = this.#buildMetaFeatures(X)
    return this.#metaModel.predict(metaX)
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
    return this.#metaModel.predictProba(metaX)
  }

  score(X, y) {
    this.#ensureFitted()
    const preds = this.predict(X)
    const yn = normalizeY(y)
    return this.#task === 'classification' ? accuracy(yn, preds) : r2Score(yn, preds)
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

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return StackingEnsemble._loadFromParts(manifest, toc, blobs)
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#baseModels) {
      for (const m of this.#baseModels) m.dispose()
    }
    if (this.#metaModel) this.#metaModel.dispose()
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

    const metaData = new Float64Array(n * this.#nMetaCols)
    for (let b = 0; b < nBase; b++) {
      if (this.#task === 'classification') {
        const proba = this.#baseModels[b].predictProba(Xn)
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < this.#nClasses; c++) {
            metaData[i * this.#nMetaCols + b * colsPerModel + c] = proba[i * this.#nClasses + c]
          }
        }
      } else {
        const preds = this.#baseModels[b].predict(Xn)
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

  static _register() {
    if (_registered) return
    _registered = true
    const loader = (manifest, toc, blobs) => {
      return StackingEnsemble._loadFromParts(manifest, toc, blobs)
    }
    register(TYPE_ID_CLS, loader)
    register(TYPE_ID_REG, loader)
  }

  static async _loadFromParts(manifest, toc, blobs) {
    const p = manifest.params
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
    for (const name of p.estimatorNames) {
      const entry = toc.find(t => t.id === name)
      if (!entry) throw new ValidationError(`No artifact for base estimator "${name}"`)
      const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
      ens.#baseModels.push(await registryLoad(blob))
    }

    // Load meta-model
    const metaEntry = toc.find(t => t.id === p.metaName)
    if (!metaEntry) throw new ValidationError(`No artifact for meta estimator "${p.metaName}"`)
    const metaBlob = blobs.subarray(metaEntry.offset, metaEntry.offset + metaEntry.length)
    ens.#metaModel = await registryLoad(metaBlob)

    ens.#fitted = true
    return ens
  }
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
