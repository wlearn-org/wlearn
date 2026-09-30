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
 *   m.fit(X, y)  // detects from y (sync)
 */

const { inferTaskKind: detectTask } = require('./task.js')
const { ValidationError, NotFittedError, DisposedError } = require('./errors.js')

// WeakMap for internal state (allows dynamic prototype methods to access inner)
const _state = new WeakMap()

const VALID_TASKS = new Set(['classification', 'regression'])

function _validateTask(task) {
  if (task !== null && !VALID_TASKS.has(task)) {
    throw new ValidationError(`Unknown task: '${task}'. Use 'classification' or 'regression'.`)
  }
  return task
}

function _taskFromInner(inner, fallback) {
  if (!inner) return fallback

  const capabilities = inner.capabilities
  if (capabilities) {
    if (capabilities.classifier === true && capabilities.regressor !== true) {
      return 'classification'
    }
    if (capabilities.regressor === true && capabilities.classifier !== true) {
      return 'regression'
    }
  }

  if (typeof inner.getParams === 'function') {
    const params = inner.getParams()
    if (params && VALID_TASKS.has(params.task)) return params.task
  }

  return fallback
}

function _get(self) {
  const s = _state.get(self)
  if (!s) throw new ValidationError('Model: invalid instance')
  return s
}

function _ensureInner(self, name) {
  const s = _get(self)
  if (s.disposed) throw new DisposedError(`${name} has been disposed.`)
  if (!s.inner || !s.fitted) throw new NotFittedError(`${name}: not fitted`)
  return s.inner
}

/**
 * Create a unified model class from a classifier and regressor class.
 *
 * @param {Function} ClassifierCls - class with static create(params)
 * @param {Function} RegressorCls - class with static create(params)
 * @param {object} [opts]
 * @param {string} [opts.name] - class name for errors
 * @param {Function} [opts.load] - async WASM loader (called in create() to pre-load)
 * @returns {Function} unified model class
 */
function createModelClass(ClassifierCls, RegressorCls, opts = {}) {
  const modelName = opts.name || 'Model'
  const loadFn = opts.load || null
  const sameClass = ClassifierCls === RegressorCls

  class UnifiedModel {
    constructor(task, params) {
      _state.set(this, {
        inner: null,
        instances: new Map(),
        task: task,
        params: params,
        fitted: false,
        fitInProgress: false,
        disposed: false,
      })
    }

    static async create(params = {}) {
      const p = { ...params }
      const task = _validateTask(p.task ?? null)
      delete p.task

      // Pre-load WASM so fit() can create inner synchronously
      if (loadFn) await loadFn()

      const m = new UnifiedModel(task, p)
      const s = _get(m)

      if (sameClass) {
        // Task-agnostic backend: one asynchronously prepared instance can be
        // configured once labels reveal the task.
        s.inner = await ClassifierCls.create(task ? { ...p, task } : { ...p })
        s.instances.set('shared', s.inner)
      } else {
        // Task-specific classes cannot be constructed synchronously from fit().
        // An explicit task only needs its selected backend. When the task must be
        // inferred later, prepare both while construction is still async.
        if (task) {
          const SelectedCls = task === 'classification' ? ClassifierCls : RegressorCls
          const selected = await SelectedCls.create({ ...p, task })
          s.instances.set(task, selected)
          s.inner = selected
        } else {
          const classifier = await ClassifierCls.create({ ...p, task: 'classification' })
          try {
            const regressor = await RegressorCls.create({ ...p, task: 'regression' })
            s.instances.set('classification', classifier)
            s.instances.set('regression', regressor)
          } catch (error) {
            if (typeof classifier.dispose === 'function') classifier.dispose()
            throw error
          }
        }
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
        if (typeof inner.getParams === 'function') {
          const params = inner.getParams()
          s.params = { ...params }
          delete s.params.task
        }
        s.task = _taskFromInner(inner, s.task)
        s.instances.set(sameClass ? 'shared' : s.task, inner)
        s.fitted = true
        return m
      } catch (_) {
        const inner = await RegressorCls.load(bytes)
        const m = new UnifiedModel('regression', {})
        const s = _get(m)
        s.inner = inner
        if (typeof inner.getParams === 'function') {
          const params = inner.getParams()
          s.params = { ...params }
          delete s.params.task
        }
        s.task = _taskFromInner(inner, s.task)
        s.instances.set(sameClass ? 'shared' : s.task, inner)
        s.fitted = true
        return m
      }
    }

    fit(X, y, fitOpts) {
      const s = _get(this)
      if (s.disposed) throw new DisposedError(`${modelName} has been disposed.`)
      if (s.fitInProgress) throw new ValidationError(`${modelName} fit is already in progress`)

      const previous = {
        task: s.task,
        inner: s.inner,
        fitted: s.fitted,
      }
      const fitTask = s.task || detectTask(y)
      const fitInner = s.instances.get(sameClass ? 'shared' : fitTask) || null
      if (!fitInner) {
        throw new ValidationError(`${modelName}: task cannot be changed on a loaded model; create a new model`)
      }

      if (typeof fitInner.setParams === 'function') {
        fitInner.setParams({ ...s.params, task: fitTask })
      }

      s.task = fitTask
      s.inner = fitInner
      s.fitted = false
      s.fitInProgress = true
      const commit = () => {
        s.fitInProgress = false
        if (s.disposed) throw new DisposedError(`${modelName} has been disposed.`)
        // Explicit backend selectors (for example objective/solver/family) are
        // authoritative. Keep the wrapper task aligned with the fitted backend.
        s.task = _taskFromInner(fitInner, fitTask)
        s.fitted = true
        return this
      }
      const fail = error => {
        s.fitInProgress = false
        s.task = previous.task
        s.inner = previous.inner
        // Preserve the fitted wrapper only when the backend explicitly reports
        // that its prior model survived. A missing/false isFitted signal is not
        // enough evidence after a possibly destructive native refit.
        s.fitted = previous.fitted && fitInner === previous.inner &&
          fitInner.isFitted === true
        throw error
      }
      try {
        const result = fitInner.fit(X, y, fitOpts)
        return result != null && typeof result.then === 'function'
          ? Promise.resolve(result).then(commit, fail)
          : commit()
      } catch (error) {
        return fail(error)
      }
    }

    predict(X, predOpts) {
      return _ensureInner(this, modelName).predict(X, predOpts)
    }

    predictProba(X, predOpts) {
      const inner = _ensureInner(this, modelName)
      if (typeof inner.predictProba !== 'function') {
        throw new ValidationError(`${modelName}: predictProba not available`)
      }
      return inner.predictProba(X, predOpts)
    }

    score(X, y, scoreOpts) {
      return _ensureInner(this, modelName).score(X, y, scoreOpts)
    }

    save() {
      return _ensureInner(this, modelName).save()
    }

    dispose() {
      const s = _get(this)
      if (s.disposed) return
      if (s.fitInProgress) {
        throw new ValidationError(`Cannot dispose ${modelName} while fit is in progress`)
      }
      const disposed = new Set()
      let firstError = null
      for (const instance of s.instances.values()) {
        if (!instance || disposed.has(instance)) continue
        disposed.add(instance)
        try {
          if (typeof instance.dispose === 'function') instance.dispose()
        } catch (error) {
          firstError ??= error
        }
      }
      s.instances.clear()
      s.inner = null
      s.fitted = false
      s.disposed = true
      if (firstError) throw firstError
    }

    getParams() {
      const s = _get(this)
      const p = s.inner && typeof s.inner.getParams === 'function'
        ? { ...s.inner.getParams() }
        : { ...s.params }
      if (s.task) p.task = s.task
      return p
    }

    setParams(p) {
      const s = _get(this)
      if (s.disposed) throw new DisposedError(`${modelName} has been disposed.`)
      if (s.fitInProgress) {
        throw new ValidationError(`Cannot set ${modelName} params while fit is in progress`)
      }
      const updates = { ...p }
      let taskChanged = false
      let newTask = s.task

      if (Object.prototype.hasOwnProperty.call(updates, 'task')) {
        newTask = _validateTask(updates.task ?? null)
        delete updates.task
        taskChanged = newTask !== s.task
        if (taskChanged && newTask && !sameClass && !s.instances.has(newTask)) {
          throw new ValidationError(`${modelName}: task cannot be changed on a loaded model; create a new model`)
        }
      }

      s.task = newTask
      s.params = { ...s.params, ...updates }

      for (const [key, instance] of s.instances) {
        if (!instance || typeof instance.setParams !== 'function') continue
        const instanceTask = sameClass ? s.task : key
        instance.setParams(instanceTask
          ? { ...s.params, task: instanceTask }
          : { ...s.params })
      }
      s.inner = sameClass
        ? s.instances.get('shared') || null
        : (s.task ? s.instances.get(s.task) || null : null)
      if (taskChanged || Object.keys(updates).length > 0) s.fitted = false
      return this
    }

    get task() { return _get(this).task }

    get isFitted() {
      const s = _get(this)
      if (!s.inner || !s.fitted) return false
      return s.inner.isFitted !== undefined ? s.inner.isFitted : true
    }

    get classes() {
      const s = _get(this)
      if (!s.inner || !s.fitted) return new Int32Array(0)
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
    // Backend capabilities may be implemented on a shared base class. Walk
    // derived-first so an override owns each forwarded method/property.
    for (let prototype = Cls.prototype; prototype !== Object.prototype && prototype; prototype = Object.getPrototypeOf(prototype)) {
      for (const key of Object.getOwnPropertyNames(prototype)) {
        if (key.startsWith('_') || key.startsWith('#')) continue
        if (standardKeys.has(key) || standardGetters.has(key)) continue
        if (key in UnifiedModel.prototype) continue

        const desc = Object.getOwnPropertyDescriptor(prototype, key)
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
  }

  // Static methods
  if (ClassifierCls.defaultSearchSpace || RegressorCls.defaultSearchSpace) {
    UnifiedModel.defaultSearchSpace = (...args) => {
      return ClassifierCls.defaultSearchSpace?.(...args) || RegressorCls.defaultSearchSpace?.(...args) || {}
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
