const { ValidationError } = require('./errors.js')
const { normalizeX, normalizeY, validateMatrix } = require('./matrix.js')

const TASK_KINDS = [
  'classification',
  'regression',
  'clustering',
  'ranking',
  'survival',
  'forecasting',
  'multioutput',
  'anomaly'
]

function inferTaskKind(y) {
  if (y == null) return 'clustering'
  const yn = normalizeY(y)
  if (yn instanceof Int32Array) return 'classification'

  let integerCount = 0
  const seen = new Set()
  for (let i = 0; i < yn.length; i++) {
    if (Number.isInteger(yn[i])) {
      integerCount++
      seen.add(yn[i])
    }
  }
  if (integerCount === yn.length && seen.size > 1 && seen.size <= 20) {
    return 'classification'
  }
  return 'regression'
}

function createFeatureSchema(X, {
  names,
  types,
  roles,
  metadata = {}
} = {}) {
  const Xn = validateMatrix(normalizeX(X))
  _validateOptionalLength(names, Xn.cols, 'names')
  _validateOptionalLength(types, Xn.cols, 'types')
  _validateOptionalLength(roles, Xn.cols, 'roles')
  const features = []
  for (let i = 0; i < Xn.cols; i++) {
    features.push({
      name: names && names[i] ? String(names[i]) : `x${i}`,
      index: i,
      type: types && types[i] ? String(types[i]) : 'numeric',
      role: roles && roles[i] ? String(roles[i]) : 'feature'
    })
  }
  return {
    rows: Xn.rows,
    cols: Xn.cols,
    features,
    metadata: { ...metadata }
  }
}

function validateFeatureSchema(schema, cols, rows) {
  if (!schema || typeof schema !== 'object') {
    throw new ValidationError('FeatureSchema must be an object')
  }
  if (!Number.isInteger(schema.rows) || schema.rows < 1) {
    throw new ValidationError('FeatureSchema.rows must be a positive integer')
  }
  if (!Number.isInteger(schema.cols) || schema.cols < 1) {
    throw new ValidationError('FeatureSchema.cols must be a positive integer')
  }
  if (rows != null && schema.rows !== rows) {
    throw new ValidationError(`FeatureSchema.rows (${schema.rows}) does not match matrix rows (${rows})`)
  }
  if (cols != null && schema.cols !== cols) {
    throw new ValidationError(`FeatureSchema.cols (${schema.cols}) does not match matrix cols (${cols})`)
  }
  if (!Array.isArray(schema.features) || schema.features.length !== schema.cols) {
    throw new ValidationError('FeatureSchema.features length must equal FeatureSchema.cols')
  }
  const names = new Set()
  for (let i = 0; i < schema.features.length; i++) {
    const f = schema.features[i]
    if (!f || typeof f !== 'object') {
      throw new ValidationError(`FeatureSchema.features[${i}] must be an object`)
    }
    if (typeof f.name !== 'string' || f.name.length === 0) {
      throw new ValidationError(`FeatureSchema.features[${i}].name must be a non-empty string`)
    }
    if (names.has(f.name)) {
      throw new ValidationError(`FeatureSchema contains duplicate feature name "${f.name}"`)
    }
    names.add(f.name)
    if (!Number.isInteger(f.index) || f.index !== i) {
      throw new ValidationError(`FeatureSchema.features[${i}].index must equal ${i}`)
    }
    if (typeof f.type !== 'string' || f.type.length === 0) {
      throw new ValidationError(`FeatureSchema.features[${i}].type must be a non-empty string`)
    }
    if (typeof f.role !== 'string' || f.role.length === 0) {
      throw new ValidationError(`FeatureSchema.features[${i}].role must be a non-empty string`)
    }
  }
  return schema
}

function createTask({
  id,
  kind,
  X,
  y,
  featureSchema,
  targetSchema,
  rowIds,
  groups,
  weights,
  rowRoles,
  provenance,
  metadata = {}
}) {
  if (X == null) throw new ValidationError('Task requires X')
  const Xn = validateMatrix(normalizeX(X))
  const yn = y == null ? undefined : normalizeY(y)
  const taskKind = kind || inferTaskKind(yn)

  if (!TASK_KINDS.includes(taskKind)) {
    throw new ValidationError(`Unsupported task kind "${taskKind}"`)
  }
  if (yn && yn.length !== Xn.rows) {
    throw new ValidationError(`Task y length (${yn.length}) must match X rows (${Xn.rows})`)
  }

  const groupValues = groups == null ? undefined : normalizeY(groups)
  if (groupValues && groupValues.length !== Xn.rows) {
    throw new ValidationError(`Task groups length (${groupValues.length}) must match X rows (${Xn.rows})`)
  }

  const weightValues = weights == null ? undefined : normalizeY(weights)
  if (weightValues && weightValues.length !== Xn.rows) {
    throw new ValidationError(`Task weights length (${weightValues.length}) must match X rows (${Xn.rows})`)
  }

  if (rowIds && rowIds.length !== Xn.rows) {
    throw new ValidationError(`Task rowIds length (${rowIds.length}) must match X rows (${Xn.rows})`)
  }

  const schema = featureSchema || createFeatureSchema(Xn)
  validateFeatureSchema(schema, Xn.cols, Xn.rows)
  if (rowRoles) validateRowRoles(rowRoles, Xn.rows)

  const task = {
    id: id || `${taskKind}-${Xn.rows}x${Xn.cols}`,
    kind: taskKind,
    X: Xn,
    featureSchema: schema,
    metadata: { ...metadata }
  }
  if (yn) task.y = yn
  if (targetSchema) task.targetSchema = { ...targetSchema }
  if (rowIds) task.rowIds = rowIds
  if (groupValues) task.groups = groupValues
  if (weightValues) task.weights = weightValues
  if (rowRoles) task.rowRoles = { ...rowRoles }
  if (provenance) task.provenance = { ...provenance }
  return validateTask(task)
}

function validateTask(task) {
  if (!task || typeof task !== 'object') {
    throw new ValidationError('Task must be an object')
  }
  if (typeof task.id !== 'string' || task.id.length === 0) {
    throw new ValidationError('Task.id must be a non-empty string')
  }
  if (!TASK_KINDS.includes(task.kind)) {
    throw new ValidationError(`Unsupported task kind "${task.kind}"`)
  }
  validateMatrix(task.X)
  validateFeatureSchema(task.featureSchema, task.X.cols, task.X.rows)
  if (task.y && task.y.length !== task.X.rows) {
    throw new ValidationError('Task.y length must match Task.X.rows')
  }
  if (task.groups && task.groups.length !== task.X.rows) {
    throw new ValidationError('Task.groups length must match Task.X.rows')
  }
  if (task.weights && task.weights.length !== task.X.rows) {
    throw new ValidationError('Task.weights length must match Task.X.rows')
  }
  if (task.rowIds && task.rowIds.length !== task.X.rows) {
    throw new ValidationError('Task.rowIds length must match Task.X.rows')
  }
  if (task.rowRoles) validateRowRoles(task.rowRoles, task.X.rows)
  return task
}

function taskRows(task) {
  validateTask(task)
  return task.X.rows
}

function validateRowRoles(rowRoles, rows) {
  if (!rowRoles || typeof rowRoles !== 'object') {
    throw new ValidationError('Task.rowRoles must be an object')
  }
  for (const [role, indices] of Object.entries(rowRoles)) {
    if (!(indices instanceof Int32Array)) {
      throw new ValidationError(`Task.rowRoles.${role} must be Int32Array`)
    }
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]
      if (!Number.isInteger(idx) || idx < 0 || idx >= rows) {
        throw new ValidationError(`Task.rowRoles.${role} contains out-of-range row index ${idx}`)
      }
    }
  }
  return rowRoles
}

function _validateOptionalLength(value, expected, name) {
  if (value != null && value.length !== expected) {
    throw new ValidationError(`FeatureSchema ${name} length (${value.length}) must equal matrix cols (${expected})`)
  }
}

module.exports = {
  TASK_KINDS,
  inferTaskKind,
  createFeatureSchema,
  validateFeatureSchema,
  validateRowRoles,
  createTask,
  validateTask,
  taskRows
}
