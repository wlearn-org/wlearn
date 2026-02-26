import {
  encodeBundle, decodeBundle, register, load as registryLoad,
  normalizeX, normalizeY, accuracy, r2Score,
  stratifiedKFold, kFold,
  ValidationError, NotFittedError, DisposedError
} from '@wlearn/core'

const TYPE_ID_CLS = 'wlearn.ensemble.bagged.classifier@1'
const TYPE_ID_REG = 'wlearn.ensemble.bagged.regressor@1'
let _registered = false

/**
 * K-fold bagged estimator with out-of-fold prediction storage.
 *
 * Trains K * nRepeats copies of a base model. Each repeat uses a different
 * seed for fold assignment. OOF predictions are accumulated (sum + count)
 * and averaged, matching AutoGluon's BaggedEnsembleModel pattern.
 */
export class BaggedEstimator {
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
  #oofCounts    // Uint8Array: per-sample prediction count
  #fitted = false
  #disposed = false

  constructor(params = {}) {
    this.#spec = params.estimator || null
    this.#kFold = params.kFold || 5
    this.#nRepeats = params.nRepeats || 1
    this.#task = params.task || 'classification'
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
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const n = Xn.rows
    this.#nSamples = n

    if (this.#task === 'classification') {
      const labelSet = new Set()
      for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
      this.#classes = new Int32Array([...labelSet].sort((a, b) => a - b))
      this.#nClasses = this.#classes.length
    }

    // Initialize OOF accumulation
    if (this.#task === 'classification') {
      this.#oofAccum = new Float64Array(n * this.#nClasses)
    } else {
      this.#oofAccum = new Float64Array(n)
    }
    this.#oofCounts = new Uint8Array(n)

    const [, EstClass, params] = this.#spec
    this.#foldModels = []

    for (let repeat = 0; repeat < this.#nRepeats; repeat++) {
      const repeatSeed = this.#seed + repeat

      const folds = this.#task === 'classification'
        ? stratifiedKFold(yn, this.#kFold, { shuffle: true, seed: repeatSeed })
        : kFold(n, this.#kFold, { shuffle: true, seed: repeatSeed })

      for (const { train, test } of folds) {
        const Xtrain = _subsetX(Xn, train)
        const ytrain = _subsetY(yn, train)
        const Xtest = _subsetX(Xn, test)

        const model = await EstClass.create(params || {})
        model.fit(Xtrain, ytrain)

        // Accumulate OOF predictions
        if (this.#task === 'classification') {
          const proba = model.predictProba(Xtest)
          const nc = this.#nClasses
          for (let i = 0; i < test.length; i++) {
            const row = test[i]
            for (let c = 0; c < nc; c++) {
              this.#oofAccum[row * nc + c] += proba[i * nc + c]
            }
          }
        } else {
          const preds = model.predict(Xtest)
          for (let i = 0; i < test.length; i++) {
            this.#oofAccum[test[i]] += preds[i]
          }
        }

        for (let i = 0; i < test.length; i++) {
          this.#oofCounts[test[i]] += 1
        }

        this.#foldModels.push(model)
      }
    }

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const Xn = normalizeX(X)
    const n = Xn.rows

    if (this.#task === 'regression') {
      const out = new Float64Array(n)
      const nModels = this.#foldModels.length
      for (const model of this.#foldModels) {
        const preds = model.predict(Xn)
        for (let i = 0; i < n; i++) out[i] += preds[i]
      }
      for (let i = 0; i < n; i++) out[i] /= nModels
      return out
    }

    // Classification: average probabilities, then argmax
    const proba = this.predictProba(Xn)
    const nc = this.#nClasses
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let bestC = 0, bestV = -Infinity
      for (let c = 0; c < nc; c++) {
        if (proba[i * nc + c] > bestV) {
          bestV = proba[i * nc + c]
          bestC = c
        }
      }
      out[i] = this.#classes[bestC]
    }
    return out
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'classification') {
      throw new ValidationError('predictProba is only available for classification')
    }

    const Xn = normalizeX(X)
    const n = Xn.rows
    const nc = this.#nClasses
    const out = new Float64Array(n * nc)
    const nModels = this.#foldModels.length

    for (const model of this.#foldModels) {
      const proba = model.predictProba(Xn)
      for (let i = 0; i < n * nc; i++) {
        out[i] += proba[i]
      }
    }
    for (let i = 0; i < n * nc; i++) {
      out[i] /= nModels
    }
    return out
  }

  score(X, y) {
    this.#ensureFitted()
    const preds = this.predict(X)
    const yn = normalizeY(y)
    return this.#task === 'classification' ? accuracy(yn, preds) : r2Score(yn, preds)
  }

  /**
   * Averaged OOF predictions.
   * Classification: flat (n * nClasses) row-major probabilities.
   * Regression: flat (n) predictions.
   */
  get oofPredictions() {
    this.#ensureFitted()
    const counts = new Uint8Array(this.#oofCounts)
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
        kFold: this.#kFold,
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

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return BaggedEstimator._loadFromParts(manifest, toc, blobs)
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#foldModels) {
      for (const m of this.#foldModels) m.dispose()
    }
    this.#foldModels = null
    this.#oofAccum = null
    this.#oofCounts = null
  }

  getParams() {
    return {
      task: this.#task,
      kFold: this.#kFold,
      nRepeats: this.#nRepeats,
      seed: this.#seed,
      estimatorName: this.#spec ? this.#spec[0] : null,
    }
  }

  setParams(p) {
    this.#ensureAlive()
    if (p.kFold !== undefined) this.#kFold = p.kFold
    if (p.nRepeats !== undefined) this.#nRepeats = p.nRepeats
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

  get isFitted() { return this.#fitted && !this.#disposed }
  get classes() { return this.#classes }

  // --- Static internals ---

  static _register() {
    if (_registered) return
    _registered = true
    const loader = (manifest, toc, blobs) =>
      BaggedEstimator._loadFromParts(manifest, toc, blobs)
    register(TYPE_ID_CLS, loader)
    register(TYPE_ID_REG, loader)
  }

  static async _loadFromParts(manifest, toc, blobs) {
    const p = manifest.params
    const bag = new BaggedEstimator({
      task: p.task,
      kFold: p.kFold || 5,
      nRepeats: p.nRepeats || 1,
      seed: p.seed ?? 42,
    })
    bag.#classes = p.classes ? new Int32Array(p.classes) : null
    bag.#nClasses = p.nClasses || 0
    bag.#nSamples = p.nSamples || 0
    bag.#spec = [p.estimatorName || 'base', null, null]

    // Load fold models
    const nFoldModels = bag.#kFold * bag.#nRepeats
    bag.#foldModels = []
    for (let i = 0; i < nFoldModels; i++) {
      const foldId = `fold_${i}`
      const entry = toc.find(t => t.id === foldId)
      if (!entry) throw new ValidationError(`No artifact for "${foldId}"`)
      const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
      bag.#foldModels.push(await registryLoad(blob))
    }

    // Load OOF data
    const oofEntry = toc.find(t => t.id === 'oof')
    if (oofEntry) {
      const oofBlob = blobs.subarray(oofEntry.offset, oofEntry.offset + oofEntry.length)
      const oof = new Float64Array(
        oofBlob.buffer.slice(oofBlob.byteOffset, oofBlob.byteOffset + oofBlob.byteLength)
      )
      bag.#oofAccum = oof
      bag.#oofCounts = new Uint8Array(bag.#nSamples).fill(1)
    } else {
      if (bag.#task === 'classification') {
        bag.#oofAccum = new Float64Array(bag.#nSamples * bag.#nClasses)
      } else {
        bag.#oofAccum = new Float64Array(bag.#nSamples)
      }
      bag.#oofCounts = new Uint8Array(bag.#nSamples)
    }

    bag.#fitted = true
    return bag
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
