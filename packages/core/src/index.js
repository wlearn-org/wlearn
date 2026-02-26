// errors
export {
  WlearnError,
  BundleError,
  RegistryError,
  ValidationError,
  NotFittedError,
  DisposedError
} from './errors.js'

// matrix
export { normalizeX, normalizeY, makeDense, validateMatrix } from './matrix.js'

// hash
export { sha256Sync } from './hash.js'

// bundle
export { encodeBundle, decodeBundle, validateBundle } from './bundle.js'

// registry
export { register, load, loadSync, getRegistry } from './registry.js'

// pipeline
export { Pipeline } from './pipeline.js'
export { Step } from './step.js'

// preprocessing
export { Preprocessor } from './preprocess.js'

// rng
export { makeLCG, shuffle } from './rng.js'

// metrics
export {
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc
} from './metrics.js'

// cross-validation
export { kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer } from './cv.js'
