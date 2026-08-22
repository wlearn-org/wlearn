'use strict'

const {
  BackendError,
  BundleError,
  CancelledError,
  DisposedError,
  NotFittedError,
  ResourceLimitError,
  StandardScaler,
  MinMaxScaler,
  ValidationError,
  encodeBundle,
  encodeJSON,
  register,
  sha256Sync,
  validateBundle
} = require('@wlearn/core')

const TYPE_ID = 'wlearn.preprocess.tabular@1'
const PLAN_MEDIA_TYPE = 'application/x-tranfi-transform-plan'
const INTERNAL = Symbol('wlearn.preprocess.internal')
const MAX_SAFE = Number.MAX_SAFE_INTEGER
const HEX_64 = /^[0-9a-f]{64}$/

const RESOLVED_KEYS = Object.freeze([
  'allMissing',
  'encode',
  'impute',
  'maxCategories',
  'maxOutputColumns',
  'maxOutputElements',
  'policyVersion',
  'scale',
  'unknownCategory'
])

const REQUEST_KEYS = new Set([
  'allMissing',
  'encode',
  'impute',
  'maxCategories',
  'maxOutputColumns',
  'maxOutputElements',
  'scale',
  'unknownCategory'
])

const DEFAULTS = Object.freeze({
  impute: 'auto',
  encode: 'auto',
  scale: false,
  maxCategories: 20,
  maxOutputColumns: 65536,
  maxOutputElements: 100000000
})

function createPreprocessAPI(defaultBackend, backendName) {
  let registrationInstalled = false
  let backendPromise = null
  let backendIdentity = null

  class Preprocessor {
    constructor(token, backend, config, runtimeOptions, effectiveLimits) {
      if (token !== INTERNAL) {
        throw new ValidationError(
          'Preprocessor construction is asynchronous. Use await Preprocessor.create(config).'
        )
      }
      this._backendModule = backend
      this._backend = backendName
      this._config = freezeJSON(config)
      this._runtimeOptions = runtimeOptions
      this._effectiveLimits = effectiveLimits
      this._plan = null
      this._inputSchema = null
      this._outputSchema = null
      this._recipeSha256 = null
      this._disposed = false
    }

    static async create(config = {}, runtimeOptions = {}) {
      const resolved = resolveConfig(config)
      const runtime = normalizeRuntimeOptions(runtimeOptions, backendName)
      const backend = await registerPreprocess()
      const limits = resolveBackendLimits(backend, runtime)
      assertConfigWithinHostLimits(resolved, limits)
      return new Preprocessor(INTERNAL, backend, resolved, runtime, limits)
    }

    static async load(bytes, runtimeOptions = {}) {
      const decoded = validateBundle(bytes)
      if (decoded.manifest.typeId !== TYPE_ID) {
        throw new ValidationError(
          `Preprocessor.load expected typeId "${TYPE_ID}", got "${decoded.manifest.typeId}"`
        )
      }
      const runtime = normalizeRuntimeOptions(runtimeOptions, backendName)
      const backend = await registerPreprocess()
      return loadFromParts(
        Preprocessor, backend, decoded.manifest, decoded.toc, decoded.blobs,
        runtime
      )
    }

    fit(X, _y) {
      this._ensureAlive()
      const matrix = normalizeMatrix(X, {
        operation: 'fit',
        allowZeroRows: false,
        expectedCols: null,
        limits: this._effectiveLimits
      })
      const schema = inputSchema(matrix.cols)
      const recipeSpec = buildRecipe(this._config, matrix.cols)
      let recipe = null
      let analyzer = null
      let plan = null
      try {
        recipe = this._backendModule.TransformRecipe.fromJSON(
          recipeSpec, { limits: this._runtimeOptions.limits }
        )
        analyzer = recipe.analyzer(schema, this._runtimeOptions)
        analyzer.push({ rows: matrix.rows, columns: matrix.columns })
        plan = analyzer.finalize()
        const input = readPlanSchema(plan, 'input', this._runtimeOptions.limits)
        const output = readPlanSchema(plan, 'output', this._runtimeOptions.limits)
        const fingerprint = plan.recipeSha256({ limits: this._runtimeOptions.limits })
        validatePlanIdentity(
          this._config, input, output, fingerprint, recipeSpec
        )
        const previous = this._plan
        this._plan = plan
        plan = null
        this._inputSchema = freezeJSON(input)
        this._outputSchema = freezeJSON(output)
        this._recipeSha256 = fingerprint
        closeQuietly(previous)
        return this
      } catch (error) {
        throw mapError(error, 'fit')
      } finally {
        closeQuietly(plan)
        closeQuietly(analyzer)
        closeQuietly(recipe)
      }
    }

    transform(X) {
      this._ensureFitted()
      const matrix = normalizeMatrix(X, {
        operation: 'transform',
        allowZeroRows: true,
        expectedCols: this._inputSchema.length,
        limits: this._effectiveLimits
      })
      let apply = null
      try {
        apply = this._plan.apply(this._inputSchema, this._runtimeOptions)
        const result = apply.run({ rows: matrix.rows, columns: matrix.columns })
        if (!result || result.rows !== matrix.rows ||
            result.columns !== this._outputSchema.length ||
            !(result.data instanceof Float64Array) ||
            result.data.length !== result.rows * result.columns) {
          throw new BackendError('Tranfi returned an invalid dense transform result.')
        }
        return {
          dtype: 'float64',
          rows: result.rows,
          cols: result.columns,
          data: new Float64Array(result.data)
        }
      } catch (error) {
        throw mapError(error, 'apply')
      } finally {
        closeQuietly(apply)
      }
    }

    fitTransform(X, y) {
      this.fit(X, y)
      return this.transform(X)
    }

    getParams() {
      this._ensureAlive()
      return cloneJSON(this._config)
    }

    setParams(params) {
      this._ensureAlive()
      const resolved = resolveSetParams(this._config, params)
      assertConfigWithinHostLimits(resolved, this._effectiveLimits)
      const oldPlan = this._plan
      this._config = freezeJSON(resolved)
      this._plan = null
      this._inputSchema = null
      this._outputSchema = null
      this._recipeSha256 = null
      closeQuietly(oldPlan)
      return this
    }

    save() {
      this._ensureFitted()
      let planBytes
      let input
      let output
      let fingerprint
      try {
        input = readPlanSchema(this._plan, 'input', this._runtimeOptions.limits)
        output = readPlanSchema(this._plan, 'output', this._runtimeOptions.limits)
        fingerprint = this._plan.recipeSha256({ limits: this._runtimeOptions.limits })
        const recipe = buildRecipe(this._config, input.length)
        validatePlanIdentity(this._config, input, output, fingerprint, recipe)
        if (!sameJSON(input, this._inputSchema) ||
            !sameJSON(output, this._outputSchema) ||
            fingerprint !== this._recipeSha256) {
          throw new BackendError('Fitted Tranfi plan identity changed unexpectedly.')
        }
        planBytes = this._plan.toBytes({ limits: this._runtimeOptions.limits })
      } catch (error) {
        throw mapError(error, 'export')
      }
      return encodeBundle({
        typeId: TYPE_ID,
        requires: [],
        params: cloneJSON(this._config),
        metadata: {
          inputSchema: cloneJSON(input),
          outputSchema: cloneJSON(output),
          tranfi: {
            abiVersion: 1,
            planFormatVersion: 1,
            recipeSha256: fingerprint
          }
        }
      }, [{ id: 'plan', mediaType: PLAN_MEDIA_TYPE, data: planBytes }])
    }

    dispose() {
      if (this._disposed) return
      this._disposed = true
      const plan = this._plan
      this._plan = null
      this._inputSchema = null
      this._outputSchema = null
      this._recipeSha256 = null
      closeQuietly(plan)
    }

    get capabilities() {
      return Object.freeze({ transformer: true })
    }

    get isFitted() {
      return !this._disposed && this._plan !== null
    }

    get backend() {
      return this._backend
    }

    get inputSchema() {
      this._ensureFitted()
      return cloneJSON(this._inputSchema)
    }

    get outputSchema() {
      this._ensureFitted()
      return cloneJSON(this._outputSchema)
    }

    _ensureAlive() {
      if (this._disposed) throw new DisposedError('Preprocessor has been disposed.')
    }

    _ensureFitted() {
      this._ensureAlive()
      if (this._plan === null) {
        throw new NotFittedError('Preprocessor is not fitted. Call fit() first.')
      }
    }
  }

  async function registerPreprocess(options = {}) {
    assertPlainObject(options, 'registerPreprocess options')
    assertExactKeys(options, ['backend'], 'registerPreprocess options', true)
    if (!registrationInstalled) {
      register(TYPE_ID, async (manifest, toc, blobs, context) => {
        const loaderOptions = context && context.loaderOptions
          ? context.loaderOptions[TYPE_ID]
          : undefined
        const runtime = normalizeRuntimeOptions(loaderOptions || {}, backendName)
        const backend = await ensureBackend()
        return loadFromParts(
          Preprocessor, backend, manifest, toc, blobs,
          runtime
        )
      }, { acceptsContext: true, sync: false })
      registrationInstalled = true
    }
    return ensureBackend(options.backend)
  }

  function ensureBackend(override) {
    const identity = override === undefined ? defaultBackend : override
    if (backendPromise !== null) {
      if (override !== undefined && identity !== backendIdentity) {
        return Promise.reject(new BackendError(
          'The preprocessing backend is already initialized for this package entry.'
        ))
      }
      return backendPromise
    }
    backendIdentity = identity
    backendPromise = Promise.resolve()
      .then(() => typeof identity === 'function' ? identity() : identity)
      .then(value => value && typeof value.then === 'function' ? value : value)
      .then(assertBackend)
      .catch(error => {
        backendPromise = null
        backendIdentity = null
        throw mapBackendInitializationError(error)
      })
    return backendPromise
  }

  return {
    TYPE_ID,
    PLAN_MEDIA_TYPE,
    Preprocessor,
    resolvePreprocessConfig,
    registerPreprocess,
    StandardScaler,
    MinMaxScaler
  }
}

/**
 * Resolve and validate a public preprocessing request without loading Tranfi.
 * AutoML uses this pure boundary before candidate IDs are created.
 */
function resolvePreprocessConfig(config = {}) {
  return freezeJSON(resolveConfig(config))
}

async function loadFromParts(Preprocessor, backend, manifest, toc, blobs, runtimeOptions) {
  validateManifestShape(manifest, toc)
  const config = resolveConfig(manifest.params, true)
  const limits = resolveBackendLimits(backend, runtimeOptions)
  assertConfigWithinHostLimits(config, limits)
  const input = validateSchema(manifest.metadata.inputSchema, 'input')
  const output = validateSchema(manifest.metadata.outputSchema, 'output')
  const expectedInput = inputSchema(input.length)
  if (!sameJSON(input, expectedInput)) {
    throw new BundleError('Preprocessor input schema is not the canonical x0..xN float64 schema.')
  }
  const expectedRecipe = buildRecipe(config, input.length)
  const expectedFingerprint = recipeFingerprint(expectedInput, expectedRecipe)
  if (manifest.metadata.tranfi.recipeSha256 !== expectedFingerprint) {
    throw new BundleError('Preprocessor params do not match the stored recipe fingerprint.')
  }
  const entry = toc[0]
  const planBytes = new Uint8Array(
    blobs.subarray(entry.offset, entry.offset + entry.length)
  )
  let plan = null
  try {
    plan = backend.TransformPlan.fromBytes(planBytes, runtimeOptions)
    const planInput = readPlanSchema(plan, 'input', runtimeOptions.limits)
    const planOutput = readPlanSchema(plan, 'output', runtimeOptions.limits)
    const fingerprint = plan.recipeSha256({ limits: runtimeOptions.limits })
    if (!sameJSON(input, planInput) || !sameJSON(output, planOutput)) {
      throw new BundleError('Preprocessor manifest schemas do not match the Tranfi plan.')
    }
    if (fingerprint !== expectedFingerprint) {
      throw new BundleError('Preprocessor manifest params do not match the Tranfi plan recipe.')
    }
    const instance = new Preprocessor(
      INTERNAL, backend, config, runtimeOptions, limits
    )
    instance._plan = plan
    plan = null
    instance._inputSchema = freezeJSON(planInput)
    instance._outputSchema = freezeJSON(planOutput)
    instance._recipeSha256 = fingerprint
    return instance
  } catch (error) {
    closeQuietly(plan)
    throw mapError(error, 'import')
  }
}

function assertBackend(backend) {
  if (!backend || typeof backend !== 'object' ||
      !backend.TransformRecipe || !backend.TransformPlan ||
      typeof backend.safeTransformLimits !== 'function') {
    throw new TypeError('Tranfi backend does not expose prepared transforms')
  }
  return backend
}

function normalizeRuntimeOptions(options, backendName) {
  if (options === undefined || options === null) options = {}
  assertPlainObject(options, 'runtime options')
  assertExactKeys(
    options, ['cancelFlag', 'cancelToken', 'limits'], 'runtime options', true
  )
  if (options.limits !== undefined) {
    assertPlainObject(options.limits, 'runtime options.limits')
  }
  if (backendName === 'native' && options.cancelToken !== undefined) {
    throw new ValidationError('The native backend uses cancelFlag, not cancelToken.')
  }
  if (backendName === 'wasm' && options.cancelFlag !== undefined) {
    throw new ValidationError('The WASM backend uses cancelToken, not cancelFlag.')
  }
  const result = {}
  if (options.limits !== undefined) result.limits = { ...options.limits }
  if (options.cancelFlag !== undefined) result.cancelFlag = options.cancelFlag
  if (options.cancelToken !== undefined) result.cancelToken = options.cancelToken
  return Object.freeze(result)
}

function resolveBackendLimits(backend, runtimeOptions) {
  try {
    return Object.freeze(backend.safeTransformLimits(runtimeOptions.limits || {}))
  } catch (error) {
    throw mapError(error, 'configuration')
  }
}

function resolveConfig(config, requireResolved = false) {
  if (config === undefined || config === null) config = {}
  assertPlainObject(config, 'preprocess config')
  const resolvedInput = isPlainObject(config.impute)
  if (requireResolved || resolvedInput || Object.prototype.hasOwnProperty.call(config, 'policyVersion')) {
    assertExactKeys(config, RESOLVED_KEYS, 'resolved preprocess config')
    return validateResolvedConfig(config)
  }
  for (const key of Object.keys(config)) {
    if (!REQUEST_KEYS.has(key)) throw new ValidationError(`Unknown preprocess config key "${key}".`)
  }
  const request = { ...DEFAULTS, ...config }
  return resolveRequestConfig(request, config)
}

function resolveSetParams(current, patch) {
  assertPlainObject(patch, 'preprocess params')
  if (isPlainObject(patch.impute) || Object.prototype.hasOwnProperty.call(patch, 'policyVersion')) {
    return resolveConfig(patch, true)
  }
  for (const key of Object.keys(patch)) {
    if (!REQUEST_KEYS.has(key)) throw new ValidationError(`Unknown preprocess config key "${key}".`)
  }
  const merged = cloneJSON(current)
  if (Object.prototype.hasOwnProperty.call(patch, 'impute')) {
    const value = patch.impute
    if (!['auto', 'mean', 'median', 'zero', false].includes(value)) {
      throw new ValidationError('impute must be auto, mean, median, zero, or false.')
    }
    merged.impute = value === false
      ? { numeric: false, categorical: false }
      : { numeric: value === 'auto' ? 'mean' : value, categorical: 'mode' }
    if (value === false) {
      if (Object.prototype.hasOwnProperty.call(patch, 'allMissing')) {
        throw new ValidationError('allMissing is invalid when imputation is disabled.')
      }
      merged.allMissing = null
    } else if (!Object.prototype.hasOwnProperty.call(patch, 'allMissing')) {
      merged.allMissing = 'zero'
    }
  }
  if (Object.prototype.hasOwnProperty.call(patch, 'encode')) {
    const value = patch.encode === 'auto' ? 'onehot' : patch.encode
    if (!['onehot', 'label', false].includes(value)) {
      throw new ValidationError('encode must be auto, onehot, label, or false.')
    }
    merged.encode = value
    if (!Object.prototype.hasOwnProperty.call(patch, 'unknownCategory')) {
      merged.unknownCategory = value === false
        ? null
        : value === 'onehot' ? 'all_zero' : 'sentinel'
    }
  }
  for (const [key, value] of Object.entries(patch)) {
    if (key !== 'impute' && key !== 'encode') merged[key] = value
  }
  return validateResolvedConfig(merged)
}

function resolveRequestConfig(request, supplied) {
  const impute = request.impute
  if (!['auto', 'mean', 'median', 'zero', false].includes(impute)) {
    throw new ValidationError('impute must be auto, mean, median, zero, or false.')
  }
  const encode = request.encode === 'auto' ? 'onehot' : request.encode
  if (!['onehot', 'label', false].includes(encode)) {
    throw new ValidationError('encode must be auto, onehot, label, or false.')
  }
  if (!['standard', 'minmax', false].includes(request.scale)) {
    throw new ValidationError('scale must be standard, minmax, or false.')
  }
  assertPositiveSafeInteger(request.maxCategories, 'maxCategories', 2)
  assertPositiveSafeInteger(request.maxOutputColumns, 'maxOutputColumns')
  assertPositiveSafeInteger(request.maxOutputElements, 'maxOutputElements')
  const imputeResolved = impute === false
    ? { numeric: false, categorical: false }
    : { numeric: impute === 'auto' ? 'mean' : impute, categorical: 'mode' }
  if (impute === false && Object.prototype.hasOwnProperty.call(supplied, 'allMissing')) {
    throw new ValidationError('allMissing is invalid when imputation is disabled.')
  }
  const allMissing = impute === false ? null : (request.allMissing ?? 'zero')
  if (allMissing !== null && !['error', 'zero'].includes(allMissing)) {
    throw new ValidationError('allMissing must be error or zero.')
  }
  if (encode === false && Object.prototype.hasOwnProperty.call(supplied, 'unknownCategory')) {
    throw new ValidationError('unknownCategory is invalid when encoding is disabled.')
  }
  const unknown = encode === false
    ? null
    : (request.unknownCategory ?? (encode === 'onehot' ? 'all_zero' : 'sentinel'))
  validateUnknownPolicy(encode, unknown)
  return validateResolvedConfig({
    impute: imputeResolved,
    encode,
    scale: request.scale,
    maxCategories: request.maxCategories,
    unknownCategory: unknown,
    allMissing,
    maxOutputColumns: request.maxOutputColumns,
    maxOutputElements: request.maxOutputElements,
    policyVersion: 1
  })
}

function validateResolvedConfig(config) {
  assertExactKeys(config, RESOLVED_KEYS, 'resolved preprocess config')
  assertPlainObject(config.impute, 'resolved preprocess config.impute')
  assertExactKeys(config.impute, ['categorical', 'numeric'], 'resolved preprocess config.impute')
  const numeric = config.impute.numeric
  const categorical = config.impute.categorical
  if (![false, 'mean', 'median', 'zero'].includes(numeric)) {
    throw new ValidationError('resolved numeric imputation is invalid.')
  }
  if (![false, 'mode'].includes(categorical) || (numeric === false) !== (categorical === false)) {
    throw new ValidationError('resolved categorical imputation is invalid or inconsistent.')
  }
  if (![false, 'onehot', 'label'].includes(config.encode)) {
    throw new ValidationError('resolved encoding is invalid.')
  }
  if (![false, 'standard', 'minmax'].includes(config.scale)) {
    throw new ValidationError('resolved scaling is invalid.')
  }
  assertPositiveSafeInteger(config.maxCategories, 'maxCategories', 2)
  assertPositiveSafeInteger(config.maxOutputColumns, 'maxOutputColumns')
  assertPositiveSafeInteger(config.maxOutputElements, 'maxOutputElements')
  if (config.policyVersion !== 1) throw new ValidationError('policyVersion must be 1.')
  if (numeric === false) {
    if (config.allMissing !== null) {
      throw new ValidationError('allMissing must be null when imputation is disabled.')
    }
  } else if (!['error', 'zero'].includes(config.allMissing)) {
    throw new ValidationError('resolved allMissing must be error or zero.')
  }
  validateUnknownPolicy(config.encode, config.unknownCategory)
  return cloneJSON(config)
}

function validateUnknownPolicy(encode, unknown) {
  if (encode === false && unknown !== null) {
    throw new ValidationError('unknownCategory must be null when encoding is disabled.')
  }
  if (encode === 'onehot' && !['error', 'all_zero'].includes(unknown)) {
    throw new ValidationError('onehot encoding requires unknownCategory error or all_zero.')
  }
  if (encode === 'label' && !['error', 'sentinel'].includes(unknown)) {
    throw new ValidationError('label encoding requires unknownCategory error or sentinel.')
  }
}

function buildRecipe(config, columns) {
  const numericImpute = config.impute.numeric === false
    ? { op: 'none', constant: null, allMissing: null }
    : config.impute.numeric === 'zero'
      ? { op: 'zero', constant: null, allMissing: null }
      : { op: config.impute.numeric, constant: null, allMissing: config.allMissing }
  const normalize = config.scale === false
    ? { op: 'none', ddof: null }
    : { op: config.scale, ddof: config.scale === 'standard' ? 0 : null }
  const categoricalImpute = config.impute.categorical === false
    ? { op: 'none', constant: null, allMissing: null }
    : { op: 'mode', constant: null, allMissing: config.allMissing }
  let categoricalEncode
  if (config.encode === false) {
    categoricalEncode = {
      op: 'none',
      categories: config.impute.categorical === false ? null : 'discover',
      unknown: null,
      sentinelLabel: null
    }
  } else {
    categoricalEncode = {
      op: config.encode,
      categories: 'discover',
      unknown: config.unknownCategory,
      sentinelLabel: config.unknownCategory === 'sentinel' ? -1 : null
    }
  }
  return {
    format: 'tranfi.transform-recipe',
    version: 1,
    policyVersion: 1,
    outputDtype: 'float64',
    semanticLimits: {
      maxOutputColumns: config.maxOutputColumns,
      maxOutputElementsPerApply: config.maxOutputElements
    },
    columns: Array.from({ length: columns }, (_, index) => ({
      sourceId: `x${index}`,
      kind: {
        op: 'infer',
        value: null,
        rule: 'finite-integer-cardinality-v1',
        maxCategories: config.maxCategories
      },
      numeric: {
        impute: { ...numericImpute },
        normalize: { ...normalize }
      },
      categorical: {
        impute: { ...categoricalImpute },
        encode: { ...categoricalEncode }
      }
    }))
  }
}

function normalizeMatrix(X, options) {
  let rows
  let cols
  let get
  if (Array.isArray(X)) {
    rows = X.length
    if (rows === 0) {
      throw new ValidationError('A zero-row matrix must declare its fitted width.')
    }
    if (!Array.isArray(X[0]) || X[0].length === 0) {
      throw new ValidationError('Matrix rows must be nonempty arrays.')
    }
    cols = X[0].length
    for (let row = 0; row < rows; row++) {
      if (!Array.isArray(X[row]) || X[row].length !== cols) {
        throw new ValidationError('Matrix must be rectangular.')
      }
    }
    get = (row, col) => X[row][col]
  } else {
    if (!isPlainObject(X)) throw new ValidationError('X must be a dense matrix or number[][].')
    rows = X.rows
    cols = X.cols
    assertNonnegativeSafeInteger(rows, 'rows')
    assertNonnegativeSafeInteger(cols, 'cols')
    const is32 = X.data instanceof Float32Array
    const is64 = X.data instanceof Float64Array
    if (!is32 && !is64) {
      throw new ValidationError('Dense matrix data must be Float32Array or Float64Array.')
    }
    const inferredDtype = is32 ? 'float32' : 'float64'
    if (X.dtype !== undefined && X.dtype !== inferredDtype) {
      throw new ValidationError('Dense matrix dtype does not match its typed array class.')
    }
    const elements = checkedProduct(rows, cols, 'rows * cols')
    if (X.data.length !== elements) {
      throw new ValidationError(`Dense matrix data length ${X.data.length} does not equal ${elements}.`)
    }
    get = (row, col) => X.data[row * cols + col]
  }
  assertNonnegativeSafeInteger(rows, 'rows')
  assertNonnegativeSafeInteger(cols, 'cols')
  if (cols === 0 || (!options.allowZeroRows && rows === 0)) {
    throw new ValidationError(`${options.operation} requires positive rows and columns.`)
  }
  if (options.expectedCols !== null && cols !== options.expectedCols) {
    throw new ValidationError(
      `Transform expected ${options.expectedCols} columns, got ${cols}.`
    )
  }
  const elements = checkedProduct(rows, cols, 'rows * cols')
  const bytes = checkedProduct(elements, 8, 'matrix byte length')
  const rowLimit = options.operation === 'fit'
    ? options.limits.maxAnalyzerRows
    : options.limits.maxApplyRows
  const byteLimit = options.operation === 'fit'
    ? options.limits.maxAnalyzerInputBytes
    : options.limits.maxApplyInputBytes
  if (cols > options.limits.maxInputColumns || rows > rowLimit || bytes > byteLimit) {
    throw new ResourceLimitError(`${options.operation} input exceeds Tranfi runtime limits.`)
  }
  const columnBytes = checkedProduct(rows, 8, 'column byte length')
  if (columnBytes > options.limits.maxAllocationBytes) {
    throw new ResourceLimitError(`${options.operation} column exceeds Tranfi allocation limit.`)
  }
  const columns = Array.from({ length: cols }, () => new Float64Array(rows))
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const value = get(row, col)
      if (typeof value !== 'number' || !Number.isFinite(value) && !Number.isNaN(value)) {
        throw new ValidationError('Matrix values must be finite numbers or NaN.')
      }
      columns[col][row] = value
    }
  }
  return { rows, cols, columns }
}

function inputSchema(columns) {
  return Array.from({ length: columns }, (_, index) => ({
    dtype: 'float64', id: `x${index}`, name: `x${index}`
  }))
}

function readPlanSchema(plan, which, limits) {
  let value
  try {
    const bytes = plan.schemaJSON(which, { limits })
    value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  } catch (error) {
    if (error && Number.isInteger(error.code)) throw error
    throw new BackendError(`Tranfi returned invalid ${which} schema JSON.`)
  }
  return validateSchema(value, which)
}

function validateSchema(schema, which) {
  if (!Array.isArray(schema) || schema.length === 0) {
    throw new BundleError(`Preprocessor ${which} schema must be a nonempty array.`)
  }
  const ids = new Set()
  return schema.map((field, index) => {
    assertPlainObject(field, `${which} schema field ${index}`, BundleError)
    const keys = which === 'input'
      ? ['dtype', 'id', 'name']
      : ['category', 'dtype', 'id', 'name', 'role', 'sourceId']
    assertExactKeys(field, keys, `${which} schema field ${index}`, false, BundleError)
    if (field.dtype !== 'float64' || typeof field.id !== 'string' || !field.id ||
        typeof field.name !== 'string' || !field.name) {
      throw new BundleError(`Preprocessor ${which} schema field ${index} is invalid.`)
    }
    if (ids.has(field.id)) throw new BundleError(`Duplicate ${which} schema id "${field.id}".`)
    ids.add(field.id)
    if (which === 'output') {
      if (typeof field.sourceId !== 'string' || !field.sourceId ||
          !['value', 'label', 'onehot'].includes(field.role)) {
        throw new BundleError(`Preprocessor output schema field ${index} metadata is invalid.`)
      }
      validateCategoryTag(field.category, index)
    }
    return cloneJSON(field)
  })
}

function validateCategoryTag(category, index) {
  if (category === null) return
  assertPlainObject(category, `output schema field ${index} category`, BundleError)
  if (category.t === 'other') {
    assertExactKeys(category, ['t'], `output schema field ${index} category`, false, BundleError)
    return
  }
  assertExactKeys(category, ['t', 'v'], `output schema field ${index} category`, false, BundleError)
  if (!['f32', 'f64'].includes(category.t) || typeof category.v !== 'string' ||
      !/^[0-9a-f]+$/.test(category.v) ||
      category.v.length !== (category.t === 'f32' ? 8 : 16)) {
    throw new BundleError(`Preprocessor output schema field ${index} category is invalid.`)
  }
}

function validateManifestShape(manifest, toc) {
  assertPlainObject(manifest, 'preprocessor manifest', BundleError)
  assertExactKeys(
    manifest,
    ['artifacts', 'bundleVersion', 'metadata', 'params', 'requires', 'typeId'],
    'preprocessor manifest', false, BundleError
  )
  if (manifest.typeId !== TYPE_ID || manifest.bundleVersion !== 1 ||
      !Array.isArray(manifest.requires) || manifest.requires.length !== 0 ||
      !Array.isArray(manifest.artifacts) || manifest.artifacts.length !== 1 ||
      !Array.isArray(toc) || toc.length !== 1) {
    throw new BundleError('Preprocessor bundle has an invalid top-level shape.')
  }
  const declaration = manifest.artifacts[0]
  const entry = toc[0]
  for (const value of [declaration, entry]) {
    if (value.id !== 'plan' || value.mediaType !== PLAN_MEDIA_TYPE) {
      throw new BundleError('Preprocessor bundle must contain exactly one Tranfi plan artifact.')
    }
  }
  if (declaration.length !== entry.length || declaration.sha256 !== entry.sha256) {
    throw new BundleError('Preprocessor artifact declaration and TOC disagree.')
  }
  assertPlainObject(manifest.metadata, 'preprocessor metadata', BundleError)
  assertExactKeys(
    manifest.metadata, ['inputSchema', 'outputSchema', 'tranfi'],
    'preprocessor metadata', false, BundleError
  )
  assertPlainObject(manifest.metadata.tranfi, 'preprocessor metadata.tranfi', BundleError)
  assertExactKeys(
    manifest.metadata.tranfi,
    ['abiVersion', 'planFormatVersion', 'recipeSha256'],
    'preprocessor metadata.tranfi', false, BundleError
  )
  const info = manifest.metadata.tranfi
  if (info.abiVersion !== 1 || info.planFormatVersion !== 1 ||
      typeof info.recipeSha256 !== 'string' || !HEX_64.test(info.recipeSha256)) {
    throw new BundleError('Preprocessor Tranfi metadata is invalid or unsupported.')
  }
}

function validatePlanIdentity(config, input, output, fingerprint, recipe) {
  const expectedInput = inputSchema(input.length)
  if (!sameJSON(input, expectedInput)) {
    throw new BackendError('Tranfi plan input schema differs from the canonical wlearn schema.')
  }
  if (output.length > config.maxOutputColumns) {
    throw new ResourceLimitError('Fitted output schema exceeds maxOutputColumns.')
  }
  const expected = recipeFingerprint(expectedInput, recipe)
  if (typeof fingerprint !== 'string' || fingerprint !== expected) {
    throw new BackendError('Tranfi plan recipe fingerprint does not match the wlearn config.')
  }
}

function recipeFingerprint(schema, recipe) {
  return sha256Sync(encodeJSON({ inputSchema: schema, policyVersion: 1, recipe }))
}

function assertConfigWithinHostLimits(config, limits) {
  if (config.maxCategories > limits.maxCategoriesPerColumn ||
      config.maxOutputColumns > limits.maxOutputColumns ||
      config.maxOutputElements > limits.maxOutputElementsPerCall) {
    throw new ResourceLimitError('Preprocess semantic limits exceed the active Tranfi host profile.')
  }
}

function mapError(error, phase) {
  if (error instanceof ValidationError || error instanceof ResourceLimitError ||
      error instanceof CancelledError || error instanceof BundleError ||
      error instanceof BackendError || error instanceof DisposedError ||
      error instanceof NotFittedError) return error
  const code = error && Number.isInteger(error.code) ? error.code : null
  if (code === null) return mapBackendInitializationError(error)
  let mapped
  if ((code >= 100 && code <= 103) || code === 107 || code === 108) {
    mapped = new ValidationError(error.message)
  } else if (code === 104) {
    mapped = new ResourceLimitError(error.message)
  } else if (code === 109) {
    mapped = new CancelledError(error.message)
  } else if (phase === 'import' && (code === 105 || code === 106)) {
    mapped = new BundleError(error.message)
  } else {
    mapped = new BackendError(error.message)
  }
  mapped.engine = 'tranfi'
  mapped.engineCode = code
  mapped.cause = error
  return mapped
}

function mapBackendInitializationError(error) {
  if (error instanceof BackendError || error instanceof ValidationError ||
      error instanceof ResourceLimitError) return error
  const mapped = new BackendError(
    `Unable to initialize the Tranfi prepared-transform backend: ${error && error.message ? error.message : error}`
  )
  mapped.engine = 'tranfi'
  mapped.engineCode = null
  mapped.cause = error
  return mapped
}

function closeQuietly(value) {
  if (!value) return
  try {
    if (typeof value.close === 'function') value.close()
    else if (typeof value.dispose === 'function') value.dispose()
  } catch (_) {
    // Preserve the primary operation result/error during deterministic cleanup.
  }
}

function assertPlainObject(value, label, ErrorClass = ValidationError) {
  if (!isPlainObject(value)) throw new ErrorClass(`${label} must be an object.`)
}

function isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function assertExactKeys(value, keys, label, allowSubset = false, ErrorClass = ValidationError) {
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  const unknown = actual.filter(key => !expected.includes(key))
  const missing = allowSubset ? [] : expected.filter(key => !actual.includes(key))
  if (unknown.length || missing.length) {
    throw new ErrorClass(`${label} must contain ${allowSubset ? 'only' : 'exactly'}: ${expected.join(', ')}.`)
  }
}

function assertPositiveSafeInteger(value, label, minimum = 1) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new ValidationError(`${label} must be a safe integer >= ${minimum}.`)
  }
}

function assertNonnegativeSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ValidationError(`${label} must be a nonnegative safe integer.`)
  }
}

function checkedProduct(left, right, label) {
  if (left !== 0 && right > Math.floor(MAX_SAFE / left)) {
    throw new ResourceLimitError(`${label} exceeds the portable safe integer range.`)
  }
  return left * right
}

function cloneJSON(value) {
  if (value === null || typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(cloneJSON)
  const copy = {}
  for (const [key, child] of Object.entries(value)) copy[key] = cloneJSON(child)
  return copy
}

function freezeJSON(value) {
  const copy = cloneJSON(value)
  if (copy && typeof copy === 'object') {
    for (const child of Object.values(copy)) freezeInPlace(child)
    Object.freeze(copy)
  }
  return copy
}

function freezeInPlace(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value)) freezeInPlace(child)
  return Object.freeze(value)
}

function sameJSON(left, right) {
  return JSON.stringify(left) === JSON.stringify(right)
}

module.exports = {
  createPreprocessAPI,
  buildRecipe,
  recipeFingerprint,
  resolveConfig,
  resolvePreprocessConfig
}
