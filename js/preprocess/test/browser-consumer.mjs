import core from '@wlearn/core'
import preprocess from '@wlearn/preprocess'

const MODEL_TYPE_ID = 'wlearn.test.browser-final@1'

class BrowserFinalModel {
  constructor(cols = null) {
    this.cols = cols
    this.fitted = cols !== null
    this.disposed = false
  }

  fit(X) {
    this.cols = X.cols
    this.fitted = true
    return this
  }

  predict(X) {
    if (!this.fitted || this.disposed) throw new Error('browser final model is unavailable')
    return new Float64Array(X.rows).fill(this.cols)
  }

  save() {
    return core.encodeBundle({
      typeId: MODEL_TYPE_ID,
      params: { cols: this.cols }
    }, [])
  }

  getParams() {
    return { cols: this.cols }
  }

  setParams(params) {
    if (params.cols !== undefined) this.cols = params.cols
    return this
  }

  dispose() {
    this.disposed = true
  }

  get capabilities() {
    return { regressor: true }
  }
}

core.register(
  MODEL_TYPE_ID,
  manifest => new BrowserFinalModel(manifest.params.cols),
  { sync: true }
)

export async function runNestedConsumer() {
  const scalerIdentity =
    preprocess.StandardScaler === core.StandardScaler &&
    preprocess.MinMaxScaler === core.MinMaxScaler
  await preprocess.registerPreprocess()
  const transformer = await preprocess.Preprocessor.create({
    impute: 'median', encode: 'label', scale: 'minmax'
  })
  const pipeline = new core.Pipeline([
    ['preprocess', transformer],
    ['model', new BrowserFinalModel()]
  ])
  pipeline.fit(
    [[1, 4.5], [2, 1.5], [1, NaN], [2, 9.5]],
    new Float64Array([0, 1, 0, 1])
  )
  const bytes = pipeline.save()
  pipeline.dispose()

  const restored = await core.load(bytes, {
    loaderOptions: {
      [preprocess.TYPE_ID]: { limits: { maxApplyRows: 2 } }
    }
  })
  try {
    const predictions = restored.predict([[2, 4.5], [3, 5.5]])
    let limitCode = null
    try {
      restored.predict([[1, 1], [2, 2], [3, 3]])
    } catch (error) {
      limitCode = error.code
    }
    return {
      bytes: bytes.byteLength,
      predictions: Array.from(predictions),
      limitCode,
      scalerIdentity
    }
  } finally {
    restored.dispose()
  }
}
