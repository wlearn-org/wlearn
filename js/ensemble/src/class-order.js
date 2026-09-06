const { ValidationError } = require('@wlearn/core')

function normalizeClassOrder(value, expectedLength, label) {
  if (!Array.isArray(value) &&
      !(ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    throw new ValidationError(`${label}: classes must be an array of unique int32 values`)
  }
  if (value.length !== expectedLength) {
    throw new ValidationError(
      `${label}: classes must contain one unique int32 label per probability column`
    )
  }
  const output = new Int32Array(expectedLength)
  const known = new Set()
  for (let index = 0; index < value.length; index++) {
    const item = value[index]
    if (typeof item !== 'number' || !Number.isInteger(item) ||
        item < -2147483648 || item > 2147483647 || known.has(item)) {
      throw new ValidationError(
        `${label}: classes must contain one unique int32 label per probability column`
      )
    }
    known.add(item)
    output[index] = item
  }
  return output
}

function classColumnMap(model, expectedClasses, label) {
  let actual = model?.classes
  if (typeof actual === 'function') actual = actual.call(model)
  if (!Array.isArray(actual) &&
      !(ArrayBuffer.isView(actual) && !(actual instanceof DataView))) {
    throw new ValidationError(`${label} must expose its fitted classes`)
  }
  if (actual.length !== expectedClasses.length) {
    throw new ValidationError(`${label} classes do not match the ensemble classes`)
  }

  const localColumns = new Map()
  for (let index = 0; index < actual.length; index++) {
    const value = actual[index]
    if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647 ||
        localColumns.has(value)) {
      throw new ValidationError(`${label} classes must be unique int32 values`)
    }
    localColumns.set(value, index)
  }

  const mapping = new Int32Array(expectedClasses.length)
  for (let index = 0; index < expectedClasses.length; index++) {
    const local = localColumns.get(expectedClasses[index])
    if (local === undefined) {
      throw new ValidationError(`${label} classes do not match the ensemble classes`)
    }
    mapping[index] = local
  }
  return mapping
}

function requireProbabilityModel(model, label) {
  if (model?.capabilities?.predictProba !== true) {
    throw new ValidationError(
      `${label} must declare predictProba capability for probability aggregation`
    )
  }
  if (typeof model?.predictProba !== 'function') {
    throw new ValidationError(`${label} must implement predictProba`)
  }
  return model
}

function validateProbabilityOutput(value, rows, classCount, label) {
  if (!Array.isArray(value) &&
      !(ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    throw new ValidationError(`${label} predictProba output must be an array`)
  }
  const expectedLength = rows * classCount
  if (!Number.isSafeInteger(expectedLength) || value.length !== expectedLength) {
    throw new ValidationError(`${label} predictProba output has the wrong shape`)
  }
  for (let index = 0; index < value.length; index++) {
    if (typeof value[index] !== 'number' || !Number.isFinite(value[index])) {
      throw new ValidationError(`${label} predictProba output must contain finite numbers`)
    }
  }
  return value
}

function validateLabelOutput(value, rows, classes, label) {
  if (!Array.isArray(value) &&
      !(ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    throw new ValidationError(`${label} predict output must be an array`)
  }
  if (value.length !== rows) {
    throw new ValidationError(`${label} predict output has the wrong shape`)
  }
  const allowed = new Set(classes)
  const output = new Int32Array(rows)
  for (let index = 0; index < rows; index++) {
    const item = value[index]
    if (typeof item !== 'number' || !Number.isInteger(item) ||
        item < -2147483648 || item > 2147483647 || !allowed.has(item)) {
      throw new ValidationError(
        `${label} predict output must contain declared int32 class labels`
      )
    }
    output[index] = item
  }
  return output
}

function validateRegressionOutput(value, rows, label) {
  if (!Array.isArray(value) &&
      !(ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    throw new ValidationError(`${label} predict output must be an array`)
  }
  if (value.length !== rows) {
    throw new ValidationError(`${label} predict output has the wrong shape`)
  }
  for (let index = 0; index < value.length; index++) {
    if (typeof value[index] !== 'number' || !Number.isFinite(value[index])) {
      throw new ValidationError(`${label} predict output must contain finite numbers`)
    }
  }
  return value
}

module.exports = {
  classColumnMap,
  normalizeClassOrder,
  requireProbabilityModel,
  validateLabelOutput,
  validateProbabilityOutput,
  validateRegressionOutput,
}
