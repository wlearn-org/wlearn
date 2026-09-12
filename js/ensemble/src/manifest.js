const { ValidationError, resolveCv } = require('@wlearn/core')

const NESTED_MEDIA_TYPE = 'application/x-wlearn-bundle'
const OOF_MEDIA_TYPE = 'application/octet-stream'

function validateVotingManifest(manifest, toc, classifierType, regressorType) {
  const p = validateCommon(manifest, classifierType, regressorType, 'VotingEnsemble')
  const names = validateNames(p.estimatorNames, 'VotingEnsemble estimatorNames')
  if (!Array.isArray(p.weights) || p.weights.length !== names.length ||
      p.weights.some(weight => typeof weight !== 'number' ||
        !Number.isFinite(weight) || weight < 0) ||
      !Number.isFinite(p.weights.reduce((sum, weight) => sum + weight, 0)) ||
      p.weights.reduce((sum, weight) => sum + weight, 0) <= 0) {
    throw new ValidationError(
      'VotingEnsemble weights must be a nonnegative finite number array with ' +
      'a positive sum matching estimatorNames'
    )
  }
  if (p.voting !== 'soft' && p.voting !== 'hard') {
    throw new ValidationError('VotingEnsemble voting must be "soft" or "hard"')
  }
  validateClasses(p, manifest.typeId === classifierType, 'VotingEnsemble')
  validateArtifacts(toc, names.map(id => [id, NESTED_MEDIA_TYPE]), 'VotingEnsemble')
  return p
}

function validateStackingManifest(manifest, toc, classifierType, regressorType) {
  const p = validateCommon(manifest, classifierType, regressorType, 'StackingEnsemble')
  const names = validateNames(p.estimatorNames, 'StackingEnsemble estimatorNames')
  const metaName = validateName(p.metaName, 'StackingEnsemble metaName')
  if (names.includes(metaName)) {
    throw new ValidationError('StackingEnsemble metaName must differ from every base estimator name')
  }
  validateCv(p.cv, 'StackingEnsemble cv')
  if (typeof p.passthrough !== 'boolean') {
    throw new ValidationError('StackingEnsemble passthrough must be a boolean')
  }
  if (!Number.isSafeInteger(p.seed)) {
    throw new ValidationError('StackingEnsemble seed must be a safe integer')
  }
  const classes = validateClasses(
    p, manifest.typeId === classifierType, 'StackingEnsemble'
  )
  assertInteger(p.nMetaCols, 1, 'StackingEnsemble nMetaCols')
  const learnedColumns = names.length * (classes === null ? 1 : classes.length)
  if ((!p.passthrough && p.nMetaCols !== learnedColumns) ||
      (p.passthrough && p.nMetaCols < learnedColumns)) {
    throw new ValidationError(
      'StackingEnsemble nMetaCols is inconsistent with its base estimators and classes'
    )
  }
  validateArtifacts(
    toc,
    [...names, metaName].map(id => [id, NESTED_MEDIA_TYPE]),
    'StackingEnsemble'
  )
  return p
}

function validateBaggingManifest(manifest, toc, classifierType, regressorType) {
  const p = validateCommon(manifest, classifierType, regressorType, 'BaggedEstimator')
  const foldCount = validateCv(p.kFold, 'BaggedEstimator kFold', p.nSamples)
  assertInteger(p.nRepeats, 1, 'BaggedEstimator nRepeats')
  if (!Number.isSafeInteger(foldCount * p.nRepeats)) {
    throw new ValidationError('BaggedEstimator fold model count exceeds the safe integer range')
  }
  if (!Number.isSafeInteger(p.seed)) {
    throw new ValidationError('BaggedEstimator seed must be a safe integer')
  }
  validateName(p.estimatorName, 'BaggedEstimator estimatorName')
  const classes = validateClasses(
    p, manifest.typeId === classifierType, 'BaggedEstimator'
  )
  assertInteger(p.nSamples, 1, 'BaggedEstimator nSamples')
  const expectedClasses = classes === null ? 0 : classes.length
  if (!Number.isSafeInteger(p.nClasses) || p.nClasses !== expectedClasses) {
    throw new ValidationError('BaggedEstimator nClasses is inconsistent with classes')
  }

  const expected = []
  const modelCount = foldCount * p.nRepeats
  const hasOof = toc.some(entry => entry.id === 'oof')
  if (toc.length !== modelCount + (hasOof ? 1 : 0)) {
    throw new ValidationError('BaggedEstimator artifact count is inconsistent with its params')
  }
  for (let index = 0; index < modelCount; index++) {
    expected.push([`fold_${index}`, NESTED_MEDIA_TYPE])
  }
  const oofEntry = toc.find(entry => entry.id === 'oof')
  if (oofEntry) {
    const valueCount = p.nSamples * (classes === null ? 1 : classes.length)
    const byteCount = valueCount * 8
    if (!Number.isSafeInteger(valueCount) || !Number.isSafeInteger(byteCount) ||
        oofEntry.length !== byteCount) {
      throw new ValidationError('BaggedEstimator OOF artifact length is inconsistent with params')
    }
    expected.push(['oof', OOF_MEDIA_TYPE])
  }
  validateArtifacts(toc, expected, 'BaggedEstimator')
  return p
}

function validateCv(cv, label, rows) {
  if (typeof cv === 'number') {
    assertInteger(cv, 2, label)
    return cv
  }
  if (!Array.isArray(cv) || !cv.length) throw new ValidationError(`${label} must be an integer or fold array`)
  if (rows == null) {
    rows = 0
    for (const fold of cv) for (const field of ['train', 'test']) {
      if (!Array.isArray(fold?.[field])) throw new ValidationError(`${label} has invalid folds`)
      for (const index of fold[field]) {
        if (!Number.isInteger(index) || index < 0 || index >= 2147483647) {
          throw new ValidationError(`${label} has invalid row indices`)
        }
        rows = Math.max(rows, index + 1)
      }
    }
  }
  // Validate indices without allocating a dataset from untrusted manifest sizes.
  resolveCv(cv, rows, { task: 'regression' })
  return cv.length
}

function validateCommon(manifest, classifierType, regressorType, label) {
  if (manifest.typeId !== classifierType && manifest.typeId !== regressorType) {
    throw new ValidationError(
      `${label}.load expected typeId "${classifierType}" or "${regressorType}", ` +
      `got "${manifest.typeId}"`
    )
  }
  const p = manifest.params
  if (!p || typeof p !== 'object' || Array.isArray(p)) {
    throw new ValidationError(`${label} manifest params must be an object`)
  }
  const expectedTask = manifest.typeId === regressorType ? 'regression' : 'classification'
  if (p.task !== expectedTask) {
    throw new ValidationError(`${label} task is inconsistent with its typeId`)
  }
  return p
}

function validateClasses(p, classification, label) {
  if (!classification) {
    if (p.classes !== null && p.classes !== undefined) {
      throw new ValidationError(`${label} regression manifest must not declare classes`)
    }
    return null
  }
  if (!Array.isArray(p.classes) || p.classes.length === 0) {
    throw new ValidationError(`${label} classification manifest must declare classes`)
  }
  const seen = new Set()
  for (const value of p.classes) {
    if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647 ||
        seen.has(value)) {
      throw new ValidationError(`${label} classes must be unique int32 values`)
    }
    seen.add(value)
  }
  return p.classes
}

function validateNames(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new ValidationError(`${label} must be a nonempty array`)
  }
  const names = value.map(name => validateName(name, `${label} entry`))
  if (new Set(names).size !== names.length) {
    throw new ValidationError(`${label} must contain unique names`)
  }
  return names
}

function validateName(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new ValidationError(`${label} must be a nonempty string`)
  }
  return value
}

function assertInteger(value, minimum, label) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new ValidationError(`${label} must be a safe integer >= ${minimum}`)
  }
}

function validateArtifacts(toc, expected, label) {
  if (toc.length !== expected.length) {
    throw new ValidationError(`${label} artifact count is inconsistent with its params`)
  }
  const actual = new Map(toc.map(entry => [entry.id, entry]))
  for (const [id, mediaType] of expected) {
    const entry = actual.get(id)
    if (!entry || entry.mediaType !== mediaType) {
      throw new ValidationError(`${label} artifact "${id}" is missing or has the wrong media type`)
    }
  }
}

module.exports = {
  validateVotingManifest,
  validateStackingManifest,
  validateBaggingManifest
}
