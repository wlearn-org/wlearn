// Core
const {
  Pipeline, load, loadSync, register,
  encodeBundle, decodeBundle, validateBundle,
  normalizeX, normalizeY,
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score, logLoss, rocAuc,
  kFold, stratifiedKFold, trainTestSplit, crossValScore,
  StandardScaler, MinMaxScaler,
  TASK_KINDS, createTask, validateTask, taskRows,
  PREDICTION_FIELDS, createPrediction, validatePrediction,
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, listMeasures, evaluateMeasure,
  evaluateMetricSet, aggregateMeasure,
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan,
  groupKFold, timeSeriesSplit, slidingWindowSplit, slidingIndexSplit,
  slidingPeriodSplit,
  TRIAL_STATUSES, Archive
} = require('@wlearn/core')

const { Preprocessor } = require('@wlearn/preprocess')

const { BasisClassifier, BasisRegressor, BasisTransformer, loadBasis } = require('@wlearn/basis')

// AutoML
const {
  autoFit, registerBayesianSearch,
  BayesianSearch, BayesianStrategy,
} = require('@wlearn/automl')

// Ensemble
const { StackingEnsemble, VotingEnsemble, BaggedEstimator } = require('@wlearn/ensemble')

// Models
const { LinearModel } = require('@wlearn/liblinear')
const { SVMModel } = require('@wlearn/libsvm')
const { XGBModel } = require('@wlearn/xgboost')
const { LGBModel } = require('@wlearn/lightgbm')
const { KNNModel } = require('@wlearn/nanoflann')
const { EBMModel } = require('@wlearn/ebm')
const { TsetlinModel } = require('@wlearn/tsetlin')
const { BARTModel } = require('@wlearn/stochtree')
const {
  XLearnLR, XLearnFM, XLearnFFM,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor,
  XLearnLRClassifier, XLearnLRRegressor
} = require('@wlearn/xlearn')
const {
  MLPModel, TabMModel, NAMModel,
  MLPClassifier, MLPRegressor,
  TabMClassifier, TabMRegressor,
  NAMClassifier, NAMRegressor
} = require('@wlearn/nn')
const { RFModel, loadRF } = require('@wlearn/rf')
const { GAMModel, loadGAM } = require('@wlearn/gam')
const {
  ClusterModel, silhouette, calinskiHarabasz,
  daviesBouldin, adjustedRand, loadCluster
} = require('@wlearn/cluster')
const {
  BayesianOptimizer, compileSpace, encodeParams, decodeParams, countFreeParams, loadBO
} = require('@wlearn/bo')

// Mitra requires onnxruntime peer dep -- optional
let MitraClassifier, MitraRegressor, registerMitraLoaders
try {
  const mitra = require('@wlearn/mitra')
  MitraClassifier = mitra.MitraClassifier
  MitraRegressor = mitra.MitraRegressor
  registerMitraLoaders = mitra.registerLoaders
} catch (error) {
  if (!isMissingOptionalMitra(error)) throw error
}

function isMissingOptionalMitra(error) {
  return Boolean(
    error &&
    error.code === 'MODULE_NOT_FOUND' &&
    typeof error.message === 'string' &&
    /Cannot find module ['"]@wlearn\/mitra['"]/.test(error.message)
  )
}

module.exports = {
  ...require('@wlearn/sym'),
  ...require('@wlearn/uncertainty'),
  // Core
  Pipeline, load, loadSync, register,
  encodeBundle, decodeBundle, validateBundle,
  normalizeX, normalizeY,
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score, logLoss, rocAuc,
  kFold, stratifiedKFold, trainTestSplit, crossValScore,
  StandardScaler, MinMaxScaler, Preprocessor,
  TASK_KINDS, createTask, validateTask, taskRows,
  PREDICTION_FIELDS, createPrediction, validatePrediction,
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, listMeasures, evaluateMeasure,
  evaluateMetricSet, aggregateMeasure,
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan,
  groupKFold, timeSeriesSplit, slidingWindowSplit, slidingIndexSplit,
  slidingPeriodSplit,
  TRIAL_STATUSES, Archive,
  // AutoML
  autoFit, registerBayesianSearch,
  // Ensemble
  StackingEnsemble, VotingEnsemble, BaggedEstimator,
  // Models
  BasisClassifier, BasisRegressor, BasisTransformer, loadBasis,
  LinearModel, SVMModel, XGBModel, LGBModel, KNNModel, EBMModel,
  TsetlinModel, BARTModel,
  XLearnLR, XLearnFM, XLearnFFM,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor,
  XLearnLRClassifier, XLearnLRRegressor,
  // NN (polygrad)
  MLPModel, TabMModel, NAMModel,
  MLPClassifier, MLPRegressor,
  TabMClassifier, TabMRegressor,
  NAMClassifier, NAMRegressor,
  // RF
  RFModel, loadRF,
  // GAM
  GAMModel, loadGAM,
  // Cluster
  ClusterModel, silhouette, calinskiHarabasz,
  daviesBouldin, adjustedRand, loadCluster,
  // BO
  BayesianOptimizer, BayesianStrategy, BayesianSearch,
  compileSpace, encodeParams, decodeParams, countFreeParams, loadBO,
  // Mitra (optional)
  MitraClassifier, MitraRegressor, registerMitraLoaders,
}
