// errors
const {
  WlearnError,
  BundleError,
  RegistryError,
  ValidationError,
  NotFittedError,
  DisposedError
} = require('./errors.js')

// matrix
const { normalizeX, normalizeY, makeDense, validateMatrix } = require('./matrix.js')

// hash
const { sha256Sync } = require('./hash.js')

// bundle
const { encodeBundle, decodeBundle, validateBundle, encodeJSON, decodeJSON } = require('./bundle.js')

// registry
const { register, load, loadSync, getRegistry } = require('./registry.js')

// pipeline
const { Pipeline } = require('./pipeline.js')
const { Step } = require('./step.js')

// preprocessing
const { Preprocessor } = require('./preprocess.js')
const { StandardScaler, MinMaxScaler } = require('./scalers.js')

// rng
const { makeLCG, shuffle } = require('./rng.js')

// metrics
const {
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc
} = require('./metrics.js')

// cross-validation
const { kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer } = require('./cv.js')

// lift (MaybePromise utilities)
const { isPromiseLike, lift } = require('./lift.js')

// model wrapper
const { createModelClass, detectTask } = require('./model.js')

module.exports = {
  // errors
  WlearnError, BundleError, RegistryError, ValidationError, NotFittedError, DisposedError,
  // matrix
  normalizeX, normalizeY, makeDense, validateMatrix,
  // hash
  sha256Sync,
  // bundle
  encodeBundle, decodeBundle, validateBundle, encodeJSON, decodeJSON,
  // registry
  register, load, loadSync, getRegistry,
  // pipeline
  Pipeline, Step,
  // preprocessing
  Preprocessor, StandardScaler, MinMaxScaler,
  // rng
  makeLCG, shuffle,
  // metrics
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc,
  // cross-validation
  kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer,
  // lift
  isPromiseLike, lift,
  // model
  createModelClass, detectTask
}
