'use strict'

const { ValidationError, sha256Sync } = require('@wlearn/core')
const { resolvePreprocessConfig } = require('@wlearn/preprocess')

const PREPROCESS_TYPE_ID = 'wlearn.preprocess.tabular@1'
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER
const UINT32_MAX = 0xffffffff
const encoder = new TextEncoder()

function createCandidate(model, params, preprocess = null) {
  if (!model || typeof model !== 'object') {
    throw new ValidationError('candidate model must be an object')
  }
  const displayName = assertString(model.displayName ?? model.name, 'model.displayName')
  const classId = assertString(model.classId, 'model.classId')
  assertParamsObject(params)
  const record = {
    model: {
      displayName,
      classId,
      params: cloneDomain(params)
    },
    preprocess: preprocess === null ? null : normalizePreprocess(preprocess)
  }
  // Validate the complete portable domain before exposing the record.
  encodeTagged(record)
  return freeze(record)
}

function normalizePreprocess(preprocess) {
  if (!preprocess || typeof preprocess !== 'object' || Array.isArray(preprocess)) {
    throw new ValidationError('candidate preprocess must be null or an object')
  }
  const templateId = assertString(preprocess.templateId, 'preprocess.templateId')
  if (preprocess.typeId !== PREPROCESS_TYPE_ID) {
    throw new ValidationError(`preprocess.typeId must be "${PREPROCESS_TYPE_ID}"`)
  }
  if (!Object.prototype.hasOwnProperty.call(preprocess, 'resolvedParams')) {
    throw new ValidationError('preprocess.resolvedParams is required')
  }
  return {
    templateId,
    typeId: PREPROCESS_TYPE_ID,
    resolvedParams: cloneDomain(
      resolvePreprocessConfig(preprocess.resolvedParams)
    )
  }
}

function candidateCanonicalBytes(candidate) {
  const normalized = createCandidate(candidate.model, candidate.model.params, candidate.preprocess)
  const identity = {
    version: 1,
    model: {
      classId: normalized.model.classId,
      params: normalized.model.params
    },
    preprocess: normalized.preprocess
  }
  return encoder.encode(writeCanonical(encodeTagged(identity)))
}

function candidateHash(candidate) {
  return sha256Sync(candidateCanonicalBytes(candidate))
}

function makeCandidateId(candidate) {
  return `wlc1_${candidateHash(candidate)}`
}

function seedFor(candidate, foldIdx, baseSeed) {
  assertUint32(baseSeed, 'baseSeed')
  if (!Number.isSafeInteger(foldIdx) || foldIdx < 0 || foldIdx >= UINT32_MAX) {
    throw new ValidationError('foldIdx must be an integer from 0 through 4294967294')
  }
  const hash = candidateHash(candidate)
  const word = (
    parseInt(hash.slice(0, 2), 16) |
    (parseInt(hash.slice(2, 4), 16) << 8) |
    (parseInt(hash.slice(4, 6), 16) << 16) |
    (parseInt(hash.slice(6, 8), 16) << 24)
  ) >>> 0
  const foldMix = Math.imul((foldIdx + 1) >>> 0, 0x9e3779b9) >>> 0
  return (baseSeed ^ word ^ foldMix) >>> 0
}

function registerCandidate(candidate, seen) {
  const bytes = candidateCanonicalBytes(candidate)
  const id = `wlc1_${sha256Sync(bytes)}`
  const canonical = bytesToHex(bytes)
  const previous = seen.get(id)
  if (previous !== undefined && previous !== canonical) {
    throw new ValidationError(`candidate identity collision for ${id}`)
  }
  seen.set(id, canonical)
  return id
}

function preprocessChoices(model) {
  const choices = model.preprocessChoices
  if (choices === undefined) return [null]
  if (!Array.isArray(choices) || choices.length === 0) {
    throw new ValidationError('model preprocessChoices must be a nonempty array')
  }
  return choices
}

function createCandidateTask(model, params, preprocess, seen) {
  const candidate = createCandidate(model, params, preprocess)
  const candidateId = registerCandidate(candidate, seen)
  let cls = model.cls
  if (preprocess !== null) {
    if (typeof model.createCandidateClass !== 'function') {
      throw new ValidationError(
        'a non-null preprocessing candidate requires a candidate class factory'
      )
    }
    cls = model.createCandidateClass(candidate)
  }
  return { candidateId, candidate, cls, params: candidate.model.params }
}

function classForCandidate(model, candidate) {
  if (candidate.preprocess === null) return model.cls
  if (typeof model.createCandidateClass !== 'function') {
    throw new ValidationError(
      'a non-null preprocessing candidate requires a candidate class factory'
    )
  }
  return model.createCandidateClass(candidate)
}

function normalizeModelSpecs(models, label = 'models') {
  if (!Array.isArray(models) || models.length === 0) {
    throw new ValidationError(`${label} must be a nonempty array`)
  }
  const classIds = new Set()
  return models.map((item, index) => {
    let spec
    if (Array.isArray(item)) {
      if (item.length < 2 || item.length > 3) {
        throw new ValidationError(`${label}[${index}] tuple must contain name, class, and optional params`)
      }
      spec = item.length === 3
        ? { name: item[0], cls: item[1], params: item[2] }
        : { name: item[0], cls: item[1] }
    } else if (item && typeof item === 'object') {
      spec = item
    } else {
      throw new ValidationError(`${label}[${index}] must be a model spec or tuple`)
    }
    const cls = spec.cls
    if (!cls || typeof cls.create !== 'function') {
      throw new ValidationError(`${label}[${index}].cls must expose create()`)
    }
    const displayName = assertString(spec.displayName ?? spec.name, `${label}[${index}].name`)
    const classId = assertString(spec.classId ?? cls.classId, `${label}[${index}].classId`)
    if (classIds.has(classId)) {
      throw new ValidationError(`duplicate model classId "${classId}"`)
    }
    classIds.add(classId)
    const params = Object.prototype.hasOwnProperty.call(spec, 'params')
      ? spec.params
      : {}
    assertParamsObject(params)
    if (Object.prototype.hasOwnProperty.call(spec, 'preprocessChoices')) {
      preprocessChoices(spec)
    }
    return Object.freeze({
      ...spec,
      name: displayName,
      displayName,
      classId,
      cls,
      params: cloneDomain(params)
    })
  })
}

function encodeTagged(value, stack = new Set()) {
  if (value === null) return ['null']
  if (typeof value === 'boolean') return ['boolean', value]
  if (typeof value === 'string') {
    validateUnicode(value, 'candidate string')
    return ['string', value]
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new ValidationError('candidate numbers must be finite')
    }
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new ValidationError('candidate integers must be within the JavaScript safe-integer domain')
    }
    const canonical = Object.is(value, -0) ? 0 : value
    const buffer = new ArrayBuffer(8)
    new DataView(buffer).setFloat64(0, canonical, false)
    return ['number', bytesToHex(new Uint8Array(buffer))]
  }
  if (!value || typeof value !== 'object') {
    throw new ValidationError('candidate values must use the portable JSON domain')
  }
  if (stack.has(value)) throw new ValidationError('candidate values must not be cyclic')
  stack.add(value)
  try {
    if (Array.isArray(value)) {
      assertDenseArray(value)
      return ['array', value.map(item => encodeTagged(item, stack))]
    }
    const prototype = Object.getPrototypeOf(value)
    if (prototype !== Object.prototype && prototype !== null) {
      throw new ValidationError('candidate objects must be plain objects')
    }
    const keys = Object.keys(value)
    keys.sort(compareUtf8)
    return ['object', keys.map(key => {
      validateUnicode(key, 'candidate object key')
      return [key, encodeTagged(value[key], stack)]
    })]
  } finally {
    stack.delete(value)
  }
}

function writeCanonical(value) {
  if (value === null) return 'null'
  if (value === true) return 'true'
  if (value === false) return 'false'
  if (typeof value === 'string') return quoteCanonical(value)
  if (Array.isArray(value)) return `[${value.map(writeCanonical).join(',')}]`
  throw new ValidationError('internal candidate canonicalization error')
}

function quoteCanonical(value) {
  validateUnicode(value, 'candidate string')
  let output = '"'
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i)
    if (code === 0x22) output += '\\"'
    else if (code === 0x5c) output += '\\\\'
    else if (code <= 0x1f) output += `\\u00${code.toString(16).padStart(2, '0')}`
    else output += value[i]
  }
  return `${output}"`
}

function compareUtf8(left, right) {
  const a = encoder.encode(left)
  const b = encoder.encode(right)
  const length = Math.min(a.length, b.length)
  for (let i = 0; i < length; i++) {
    if (a[i] !== b[i]) return a[i] - b[i]
  }
  return a.length - b.length
}

function validateUnicode(value, label) {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i)
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(i + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new ValidationError(`${label} contains an unpaired UTF-16 surrogate`)
      }
      i++
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new ValidationError(`${label} contains an unpaired UTF-16 surrogate`)
    }
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new ValidationError(`${label} must be a nonempty string`)
  }
  validateUnicode(value, label)
  return value
}

function assertUint32(value, label) {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new ValidationError(`${label} must be an unsigned 32-bit integer`)
  }
}

function cloneDomain(value, stack = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number') {
    return value
  }
  if (!value || typeof value !== 'object') {
    throw new ValidationError('candidate values must use the portable JSON domain')
  }
  if (stack.has(value)) throw new ValidationError('candidate values must not be cyclic')
  stack.add(value)
  try {
    if (Array.isArray(value)) {
      assertDenseArray(value)
      return value.map(item => cloneDomain(item, stack))
    }
    const prototype = Object.getPrototypeOf(value)
    if (prototype !== Object.prototype && prototype !== null) {
      throw new ValidationError('candidate objects must be plain objects')
    }
    const result = {}
    for (const [key, item] of Object.entries(value)) {
      Object.defineProperty(result, key, {
        value: cloneDomain(item, stack),
        enumerable: true,
        configurable: true,
        writable: true,
      })
    }
    return result
  } finally {
    stack.delete(value)
  }
}

function assertParamsObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ValidationError('candidate model.params must be a plain object')
  }
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) {
    throw new ValidationError('candidate model.params must be a plain object')
  }
}

function assertDenseArray(value) {
  for (let index = 0; index < value.length; index++) {
    if (!Object.prototype.hasOwnProperty.call(value, index)) {
      throw new ValidationError('candidate arrays must not be sparse')
    }
  }
}

function freeze(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value)) freeze(child)
  return Object.freeze(value)
}

function bytesToHex(bytes) {
  let result = ''
  for (const byte of bytes) result += byte.toString(16).padStart(2, '0')
  return result
}

module.exports = {
  PREPROCESS_TYPE_ID,
  candidateCanonicalBytes,
  candidateHash,
  classForCandidate,
  createCandidate,
  createCandidateTask,
  makeCandidateId,
  normalizeModelSpecs,
  preprocessChoices,
  registerCandidate,
  seedFor
}
