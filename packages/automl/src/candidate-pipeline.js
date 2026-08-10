'use strict'

const { Pipeline, normalizeX, normalizeY } = require('@wlearn/core')
const { Preprocessor } = require('@wlearn/preprocess')
const {
  classForCandidate, makeCandidateId, seedFor
} = require('./candidate.js')

/**
 * Build the estimator class for one resolved preprocessing candidate.
 *
 * Construction is deliberately sequential because each successful child owns
 * resources which must be released if a later stage fails.
 */
function createCandidatePipelineClass(spec, candidate, dependencies = {}) {
  const Model = spec.cls
  const PreprocessorClass = dependencies.Preprocessor || Preprocessor
  const PipelineClass = dependencies.Pipeline || Pipeline
  const resolvedConfig = candidate.preprocess.resolvedParams
  const provenance = candidateProvenance(candidate, dependencies)

  class PreprocessedModel {
    static async create(params = {}) {
      const preprocessor = await PreprocessorClass.create(resolvedConfig)
      let estimator = null
      try {
        estimator = await Model.create(params)
        return new PipelineClass([
          ['preprocess', preprocessor],
          ['model', estimator],
        ], { provenance })
      } catch (error) {
        disposeQuietly(estimator)
        disposeQuietly(preprocessor)
        throw error
      }
    }

    static defaultSearchSpace(...args) {
      return Model.defaultSearchSpace?.(...args) || {}
    }

    static budgetSpec(...args) {
      return Model.budgetSpec?.(...args)
    }
  }

  Object.defineProperty(PreprocessedModel, 'name', {
    value: `${Model.name || spec.name}WithPreprocessing`,
    configurable: true,
  })
  Object.defineProperty(PreprocessedModel, 'classId', {
    value: spec.classId,
    configurable: true,
  })

  return PreprocessedModel
}

function candidateProvenance(candidate, dependencies) {
  const candidateId = makeCandidateId(candidate)
  const provenance = { candidateId, candidate }
  if (dependencies.baseSeed === undefined && dependencies.foldCount === undefined) {
    return provenance
  }
  const baseSeed = dependencies.baseSeed ?? 42
  const foldCount = dependencies.foldCount ?? 5
  if (!Number.isSafeInteger(foldCount) || foldCount < 1) {
    const { ValidationError } = require('@wlearn/core')
    throw new ValidationError('foldCount must be a positive safe integer')
  }
  provenance.baseSeed = baseSeed
  provenance.foldSeeds = Array.from({ length: foldCount }, (_unused, foldId) => ({
    foldId,
    seed: seedFor(candidate, foldId, baseSeed),
  }))
  return provenance
}

function disposeQuietly(value) {
  if (!value || typeof value.dispose !== 'function') return
  try { value.dispose() } catch {}
}

async function fitCandidate(spec, candidate, X, y, candidateId = null) {
  if (candidateId !== null && makeCandidateId(candidate) !== candidateId) {
    const { ValidationError } = require('@wlearn/core')
    throw new ValidationError('candidateId does not match the structured candidate')
  }
  const CandidateClass = classForCandidate(spec, candidate)
  const instance = await CandidateClass.create(candidate.model.params)
  try {
    instance.fit(normalizeX(X), normalizeY(y))
    return instance
  } catch (error) {
    disposeQuietly(instance)
    throw error
  }
}

module.exports = { createCandidatePipelineClass, fitCandidate }
