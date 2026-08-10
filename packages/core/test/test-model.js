const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { createModelClass, detectTask } = require('../src/index.js')

// --- Mock estimator classes ---

class MockClassifier {
  #params
  #fitted = false
  #classes = null
  constructor(params) { this.#params = { ...params } }
  static async create(params) { return new MockClassifier(params) }
  fit(X, y) {
    this.#fitted = true
    const unique = new Set(y)
    this.#classes = new Int32Array([...unique].sort())
    return this
  }
  predict(X) { return new Int32Array(X.length || 3) }
  predictProba(X) { return new Float64Array((X.length || 3) * 2) }
  score(X, y) { return 0.9 }
  save() { return new Uint8Array([1, 2, 3]) }
  dispose() { this.#fitted = false }
  getParams() { return { ...this.#params } }
  setParams(p) { this.#params = { ...p }; return this }
  get isFitted() { return this.#fitted }
  get classes() { return this.#classes || new Int32Array(0) }
  // Extra method: model-specific
  explain(X) { return { importances: [0.5, 0.3, 0.2] } }
  static defaultSearchSpace() {
    return { hidden_sizes: { type: 'categorical', values: [[64], [128]] } }
  }
}

class MockRegressor {
  #params
  #fitted = false
  constructor(params) { this.#params = { ...params } }
  static async create(params) { return new MockRegressor(params) }
  fit(X, y) { this.#fitted = true; return this }
  predict(X) { return new Float64Array(X.length || 3) }
  score(X, y) { return 0.85 }
  save() { return new Uint8Array([4, 5, 6]) }
  dispose() { this.#fitted = false }
  getParams() { return { ...this.#params } }
  setParams(p) { this.#params = { ...p }; return this }
  get isFitted() { return this.#fitted }
  // Extra method (different from classifier)
  residuals() { return [0.1, -0.2, 0.05] }
  static defaultSearchSpace() {
    return { hidden_sizes: { type: 'categorical', values: [[64], [128]] } }
  }
}

class MockTaskAgnostic {
  #params
  #fitted = false
  constructor(params) { this.#params = { ...params } }
  static async create(params) { return new MockTaskAgnostic(params) }
  fit(X, y) { this.#fitted = true; return this }
  predict(X) {
    return this.#params.task === 'classification'
      ? new Int32Array(3)
      : new Float64Array(3)
  }
  score(X, y) { return 0.88 }
  save() { return new Uint8Array([7, 8, 9]) }
  dispose() { this.#fitted = false }
  getParams() { return { ...this.#params } }
  setParams(p) { this.#params = { ...p }; return this }
  get isFitted() { return this.#fitted }
  featureImportances() { return [0.4, 0.6] }
  static defaultSearchSpace() { return { depth: { type: 'int', low: 3, high: 10 } } }
}

// --- Tests ---

describe('detectTask', () => {
  it('Int32Array -> classification', () => {
    assert.equal(detectTask(new Int32Array([0, 1, 0, 1])), 'classification')
  })

  it('few unique integers -> classification', () => {
    assert.equal(detectTask([0, 1, 2, 0, 1, 2]), 'classification')
  })

  it('non-integer values -> regression', () => {
    assert.equal(detectTask([1.5, 2.3, 3.7]), 'regression')
  })

  it('many unique integers -> regression', () => {
    const y = Array.from({ length: 100 }, (_, i) => i)
    assert.equal(detectTask(y), 'regression')
  })

  it('Float64Array with few unique ints -> classification', () => {
    assert.equal(detectTask(new Float64Array([0, 1, 0, 1, 0])), 'classification')
  })
})

describe('createModelClass with classifier/regressor pair', () => {
  const MLPModel = createModelClass(MockClassifier, MockRegressor, { name: 'MLPModel' })

  it('returned class has correct name', () => {
    assert.equal(MLPModel.name, 'MLPModel')
  })

  it('create with explicit task=classification', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64], task: 'classification' })
    assert.equal(m.task, 'classification')
    await m.fit([[1, 2], [3, 4], [5, 6]], new Int32Array([0, 1, 0]))
    const pred = m.predict([[1, 2]])
    assert.ok(pred instanceof Int32Array)
    m.dispose()
  })

  it('create with explicit task=regression', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64], task: 'regression' })
    assert.equal(m.task, 'regression')
    await m.fit([[1, 2], [3, 4], [5, 6]], [1.5, 2.5, 3.5])
    const pred = m.predict([[1, 2]])
    assert.ok(pred instanceof Float64Array)
    m.dispose()
  })

  it('auto-detect classification from Int32Array labels', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64] })
    assert.equal(m.task, null)
    await m.fit([[1, 2], [3, 4], [5, 6]], new Int32Array([0, 1, 0]))
    assert.equal(m.task, 'classification')
    m.dispose()
  })

  it('auto-detect regression from float labels', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64] })
    await m.fit([[1, 2], [3, 4], [5, 6]], [1.5, 2.5, 3.5])
    assert.equal(m.task, 'regression')
    m.dispose()
  })

  it('defaultSearchSpace returns search space', () => {
    const space = MLPModel.defaultSearchSpace()
    assert.ok(space.hidden_sizes)
  })

  it('getParams includes task', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64], task: 'classification' })
    const p = m.getParams()
    assert.equal(p.task, 'classification')
    m.dispose()
  })

  it('setParams changes task, invalidates fit, and preserves params', async () => {
    // Creating without an explicit task prepares both async factories and permits
    // later synchronous task switching.
    const m = await MLPModel.create({ hidden_sizes: [64] })
    await m.fit([[1, 2]], new Int32Array([0]))
    assert.equal(m.isFitted, true)
    m.setParams({ task: 'regression', learning_rate: 0.1 })
    assert.equal(m.task, 'regression')
    assert.equal(m.isFitted, false)
    assert.throws(() => m.predict([[1, 2]]), /not fitted/)
    assert.deepEqual(m.getParams(), {
      hidden_sizes: [64],
      learning_rate: 0.1,
      task: 'regression',
    })
    m.fit([[1, 2]], new Float64Array([1.5]))
    assert.equal(m.isFitted, true)
    assert.ok(m.predict([[1, 2]]) instanceof Float64Array)
    m.dispose()
  })

  it('setParams merges updates without erasing existing params', async () => {
    const m = await MLPModel.create({ hidden_sizes: [64], task: 'classification' })
    m.setParams({ learning_rate: 0.1 })
    assert.deepEqual(m.getParams(), {
      hidden_sizes: [64],
      learning_rate: 0.1,
      task: 'classification',
    })
    m.dispose()
  })

  it('rejects invalid task values during create and setParams', async () => {
    await assert.rejects(() => MLPModel.create({ task: 'clustering' }), /Unknown task/)
    const m = await MLPModel.create({ task: 'classification' })
    assert.throws(() => m.setParams({ task: 'clustering' }), /Unknown task/)
    m.dispose()
  })

  it('score delegates to inner', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    await m.fit([[1, 2]], new Int32Array([0]))
    assert.equal(m.score([[1, 2]], new Int32Array([0])), 0.9)
    m.dispose()
  })

  it('save delegates to inner', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    await m.fit([[1, 2]], new Int32Array([0]))
    const bytes = m.save()
    assert.deepEqual(bytes, new Uint8Array([1, 2, 3]))
    m.dispose()
  })

  it('predictProba works for classifier', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    await m.fit([[1, 2], [3, 4]], new Int32Array([0, 1]))
    const proba = m.predictProba([[1, 2]])
    assert.ok(proba instanceof Float64Array)
    m.dispose()
  })

  it('predictProba throws for regressor', async () => {
    const m = await MLPModel.create({ task: 'regression' })
    await m.fit([[1, 2]], [1.5])
    assert.throws(() => m.predictProba([[1, 2]]), /not available/)
    m.dispose()
  })

  it('throws on predict before fit (no task)', async () => {
    const m = await MLPModel.create({})
    assert.throws(() => m.predict([[1, 2]]), /not fitted/)
    m.dispose()
  })

  it('isFitted reflects inner state', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    assert.equal(m.isFitted, false)
    await m.fit([[1, 2]], new Int32Array([0]))
    assert.equal(m.isFitted, true)
    m.dispose()
    assert.equal(m.isFitted, false)
  })

  it('classes proxied from inner', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    await m.fit([[1, 2], [3, 4]], new Int32Array([0, 1]))
    assert.deepEqual(m.classes, new Int32Array([0, 1]))
    m.dispose()
  })
})

describe('extra methods proxied from inner classes', () => {
  const MLPModel = createModelClass(MockClassifier, MockRegressor, { name: 'MLPModel' })

  it('classifier extra method (explain) is proxied', async () => {
    const m = await MLPModel.create({ task: 'classification' })
    await m.fit([[1, 2]], new Int32Array([0]))
    const result = m.explain([[1, 2]])
    assert.deepEqual(result, { importances: [0.5, 0.3, 0.2] })
    m.dispose()
  })

  it('regressor extra method (residuals) is proxied', async () => {
    const m = await MLPModel.create({ task: 'regression' })
    await m.fit([[1, 2]], [1.5])
    const result = m.residuals()
    assert.deepEqual(result, [0.1, -0.2, 0.05])
    m.dispose()
  })

  it('extra method throws if not fitted', async () => {
    const m = await MLPModel.create({})
    assert.throws(() => m.explain([[1, 2]]), /not fitted/)
    m.dispose()
  })
})

describe('createModelClass with single task-agnostic class', () => {
  const XGB = createModelClass(MockTaskAgnostic, MockTaskAgnostic, { name: 'XGBModel' })

  it('create and fit without task param', async () => {
    const m = await XGB.create({ max_depth: 6 })
    await m.fit([[1, 2], [3, 4]], [1.5, 2.5])
    assert.equal(m.task, 'regression')
    assert.ok(m.predict([[1, 2]]) instanceof Float64Array)
    m.dispose()
  })

  it('passes an inferred classification task to the prepared inner model', async () => {
    const m = await XGB.create({ max_depth: 6 })
    m.fit([[1, 2], [3, 4]], new Int32Array([0, 1]))
    assert.equal(m.task, 'classification')
    assert.equal(m.getParams().task, 'classification')
    assert.ok(m.predict([[1, 2]]) instanceof Int32Array)
    m.dispose()
  })

  it('can switch task, refit, and remain usable', async () => {
    const m = await XGB.create({ max_depth: 6, task: 'classification' })
    m.fit([[1, 2], [3, 4]], new Int32Array([0, 1]))
    m.setParams({ task: 'regression', eta: 0.1 })
    assert.equal(m.isFitted, false)
    assert.deepEqual(m.getParams(), {
      max_depth: 6,
      eta: 0.1,
      task: 'regression',
    })
    m.fit([[1, 2], [3, 4]], new Float64Array([1.5, 2.5]))
    assert.ok(m.predict([[1, 2]]) instanceof Float64Array)
    m.dispose()
  })

  it('extra method (featureImportances) proxied', async () => {
    const m = await XGB.create({ task: 'classification' })
    await m.fit([[1, 2]], new Int32Array([0]))
    assert.deepEqual(m.featureImportances(), [0.4, 0.6])
    m.dispose()
  })

  it('defaultSearchSpace returns class space', () => {
    const space = XGB.defaultSearchSpace()
    assert.ok(space.depth)
  })
})

describe('createModelClass edge cases', () => {
  it('works with no opts', () => {
    const M = createModelClass(MockClassifier, MockRegressor)
    assert.equal(M.name, 'Model')
  })

  it('task param passed through to inner create()', async () => {
    // Simulates task-agnostic models like XGBModel that use task param internally
    let receivedParams = null
    class TaskAware {
      static async create(params) {
        receivedParams = { ...params }
        return new TaskAware()
      }
      fit() { return this }
      predict() { return new Float64Array(1) }
      dispose() {}
    }
    const M = createModelClass(TaskAware, TaskAware)
    const m = await M.create({ max_depth: 6, task: 'classification' })
    assert.equal(receivedParams.task, 'classification')
    assert.equal(receivedParams.max_depth, 6)
    m.dispose()
  })

  it('params without task pass through to inner', async () => {
    const M = createModelClass(MockClassifier, MockRegressor)
    const m = await M.create({ lr: 0.01, task: 'classification' })
    const p = m.getParams()
    assert.equal(p.lr, 0.01)
    assert.equal(p.task, 'classification')
    m.dispose()
  })

  it('prepares task-specific factory instances instead of calling constructors from fit', async () => {
    const token = Symbol('factory')
    class FactoryClassifier {
      #params
      #fitted = false
      constructor(factoryToken, params) {
        assert.equal(factoryToken, token)
        this.#params = { ...params }
      }
      static async create(params) { return new FactoryClassifier(token, params) }
      fit() { this.#fitted = true; return this }
      predict() { return new Int32Array([1]) }
      save() { return new Uint8Array([1]) }
      dispose() { this.#fitted = false }
      getParams() { return { ...this.#params } }
      setParams(params) { this.#params = { ...params }; return this }
      get isFitted() { return this.#fitted }
      get capabilities() { return { classifier: true, regressor: false } }
    }
    class FactoryRegressor extends FactoryClassifier {
      static async create(params) { return new FactoryRegressor(token, params) }
      predict() { return new Float64Array([1.5]) }
      get capabilities() { return { classifier: false, regressor: true } }
    }

    const M = createModelClass(FactoryClassifier, FactoryRegressor)
    const m = await M.create({ alpha: 0.5 })
    m.fit([[1]], new Float64Array([1.5]))
    assert.equal(m.task, 'regression')
    assert.ok(m.predict([[1]]) instanceof Float64Array)
    m.dispose()
  })

  it('constructs only the selected factory for an explicit task', async () => {
    let classifierCreates = 0
    let regressorCreates = 0
    class AvailableClassifier extends MockClassifier {
      static async create(params) {
        classifierCreates++
        return new AvailableClassifier(params)
      }
    }
    class UnavailableRegressor extends MockRegressor {
      static async create() {
        regressorCreates++
        throw new Error('regressor backend unavailable')
      }
    }

    const M = createModelClass(AvailableClassifier, UnavailableRegressor)
    const m = await M.create({ task: 'classification' })
    assert.equal(classifierCreates, 1)
    assert.equal(regressorCreates, 0)
    m.fit([[1]], new Int32Array([0]))
    assert.ok(m.predict([[1]]) instanceof Int32Array)
    m.dispose()
  })

  it('requires a new explicit model to switch to an unprepared task', async () => {
    const M = createModelClass(MockClassifier, MockRegressor)
    const m = await M.create({ task: 'classification' })
    assert.throws(
      () => m.setParams({ task: 'regression' }),
      /task cannot be changed/
    )
    m.dispose()
  })
})
