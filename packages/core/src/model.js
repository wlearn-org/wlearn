/**
 * createModelClass(ClassifierCls, RegressorCls) -> unified model class
 *
 * Returns a class that:
 * - Accepts an optional `task` param ('classification' | 'regression')
 * - Auto-detects task from y at fit() time if not specified
 * - Creates the right inner class and proxies all calls to it
 *
 * Used INSIDE model packages to export a single unified class:
 *
 *   // nn/src/index.js
 *   const MLPModel = createModelClass(MLPClassifier, MLPRegressor)
 *   module.exports = { MLPModel }
 *
 *   // xgboost-wasm/src/index.js (already task-agnostic, pass same class twice)
 *   const XGBModel = createModelClass(XGBModelImpl, XGBModelImpl)
 *   module.exports = { XGBModel }
 *
 * End user:
 *   const m = await MLPModel.create({ hidden_sizes: [64], task: 'classification' })
 *   // or auto-detect:
 *   const m = await MLPModel.create({ hidden_sizes: [64] })
 *   await m.fit(X, y)  // detects from y
 */

const { normalizeY } = require('./matrix.js')

const { round } = Math

/**
 * Detect task type from labels.
 */
function detectTask(y) {
  const yn = normalizeY(y)
  if (yn instanceof Int32Array) return 'classification'
  const unique = new Set()
  for (let i = 0; i < yn.length; i++) {
    if (yn[i] !== round(yn[i])) return 'regression'
    unique.add(yn[i])
  }
  return unique.size <= 20 ? 'classification' : 'regression'
}

// WeakMap for internal state (allows dynamic prototype methods to access inner)
const _state = new WeakMap()

function _get(self) {
  const s = _state.get(self)
  if (!s) throw new Error('Model: invalid instance')
  return s
}

function _ensureInner(self, name) {
  const s = _get(self)
  if (!s.inner) throw new Error(`${name}: not fitted`)
  return s.inner
}

/**
 * Create a unified model class from a classifier and regressor class.
 *
 * @param {Function} ClassifierCls - class with static create(params)
 * @param {Function} RegressorCls - class with static create(params)
 * @param {object} [opts]
 * @param {string} [opts.name] - class name for errors
 * @returns {Function} unified model class
 */
function createModelClass(ClassifierCls, RegressorCls, opts = {}) {
  const modelName = opts.name || 'Model'

  class UnifiedModel {
    constructor(task, params) {
      _state.set(this, {
        inner: null,
        task: task,
        params: params,
      })
    }

    static async create(params = {}) {
      const p = { ...params }
      const task = p.task || null
      delete p.task

      const m = new UnifiedModel(task, p)

      // If task is known, create inner immediately
      if (task) {
        const cls = task === 'classification' ? ClassifierCls : RegressorCls
        const s = _get(m)
        s.inner = await cls.create({ ...p, task })
      }

      return m
    }

    static async load(bytes) {
      // Try classifier first, fall back to regressor
      try {
        const inner = await ClassifierCls.load(bytes)
        const m = new UnifiedModel('classification', {})
        const s = _get(m)
        s.inner = inner
        if (typeof inner.getParams === 'function') s.params = inner.getParams()
        return m
      } catch (_) {
        const inner = await RegressorCls.load(bytes)
        const m = new UnifiedModel('regression', {})
        const s = _get(m)
        s.inner = inner
        if (typeof inner.getParams === 'function') s.params = inner.getParams()
        return m
      }
    }

    async fit(X, y, opts) {
      const s = _get(this)

      // Auto-detect task if not set
      if (!s.task) {
        s.task = detectTask(y)
      }

      // Create inner if not yet created
      if (!s.inner) {
        const cls = s.task === 'classification' ? ClassifierCls : RegressorCls
        s.inner = await cls.create({ ...s.params, task: s.task })
      }

      s.inner.fit(X, y, opts)
      return this
    }

    predict(X, opts) {
      return _ensureInner(this, modelName).predict(X, opts)
    }

    predictProba(X, opts) {
      const inner = _ensureInner(this, modelName)
      if (typeof inner.predictProba !== 'function') {
        throw new Error(`${modelName}: predictProba not available`)
      }
      return inner.predictProba(X, opts)
    }

    score(X, y, opts) {
      return _ensureInner(this, modelName).score(X, y, opts)
    }

    save() {
      return _ensureInner(this, modelName).save()
    }

    dispose() {
      const s = _get(this)
      if (s.inner && typeof s.inner.dispose === 'function') {
        s.inner.dispose()
      }
      s.inner = null
    }

    getParams() {
      const s = _get(this)
      const p = s.inner && typeof s.inner.getParams === 'function'
        ? s.inner.getParams()
        : { ...s.params }
      if (s.task) p.task = s.task
      return p
    }

    setParams(p) {
      const s = _get(this)
      const params = { ...p }
      if (params.task) {
        const newTask = params.task
        delete params.task
        if (newTask !== s.task && s.inner) {
          this.dispose()
        }
        s.task = newTask
      }
      s.params = params
      if (s.inner && typeof s.inner.setParams === 'function') {
        s.inner.setParams(params)
      }
      return this
    }

    get task() { return _get(this).task }

    get isFitted() {
      const s = _get(this)
      if (!s.inner) return false
      return s.inner.isFitted !== undefined ? s.inner.isFitted : true
    }

    get classes() {
      const s = _get(this)
      if (!s.inner) return new Int32Array(0)
      if (typeof s.inner.classes === 'function') return s.inner.classes()
      if (s.inner.classes !== undefined) return s.inner.classes
      return new Int32Array(0)
    }

    get capabilities() {
      const s = _get(this)
      if (s.inner && s.inner.capabilities !== undefined) return s.inner.capabilities
      return {}
    }
  }

  // Discover extra methods/getters from both classes and add proxies
  const standardKeys = new Set([
    'constructor', 'fit', 'predict', 'predictProba', 'score', 'save',
    'dispose', 'getParams', 'setParams',
  ])
  const standardGetters = new Set(['classes', 'task', 'isFitted', 'capabilities'])

  for (const Cls of [ClassifierCls, RegressorCls]) {
    if (!Cls || !Cls.prototype) continue
    for (const key of Object.getOwnPropertyNames(Cls.prototype)) {
      if (key.startsWith('_') || key.startsWith('#')) continue
      if (standardKeys.has(key) || standardGetters.has(key)) continue
      if (key in UnifiedModel.prototype) continue

      const desc = Object.getOwnPropertyDescriptor(Cls.prototype, key)
      if (!desc) continue

      if (typeof desc.value === 'function') {
        UnifiedModel.prototype[key] = function (...args) {
          return _ensureInner(this, modelName)[key](...args)
        }
      } else if (desc.get) {
        Object.defineProperty(UnifiedModel.prototype, key, {
          get() {
            const s = _get(this)
            if (!s.inner) return undefined
            return s.inner[key]
          },
          configurable: true,
        })
      }
    }
  }

  // Static methods
  if (ClassifierCls.defaultSearchSpace || RegressorCls.defaultSearchSpace) {
    UnifiedModel.defaultSearchSpace = () => {
      return ClassifierCls.defaultSearchSpace?.() || RegressorCls.defaultSearchSpace?.() || {}
    }
  }

  if (ClassifierCls.budgetSpec || RegressorCls.budgetSpec) {
    UnifiedModel.budgetSpec = () => {
      return ClassifierCls.budgetSpec?.() || RegressorCls.budgetSpec?.() || undefined
    }
  }

  Object.defineProperty(UnifiedModel, 'name', { value: modelName, configurable: true })

  return UnifiedModel
}

module.exports = { createModelClass, detectTask }
