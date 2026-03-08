// Core
const {
  Pipeline, load, loadSync, register,
  encodeBundle, decodeBundle, validateBundle,
  normalizeX, normalizeY,
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score, logLoss, rocAuc,
  kFold, stratifiedKFold, trainTestSplit, crossValScore,
  StandardScaler, MinMaxScaler, Preprocessor
} = require('@wlearn/core')

// AutoML
const { autoFit } = require('@wlearn/automl')

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
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor,
  XLearnLRClassifier, XLearnLRRegressor
} = require('@wlearn/xlearn')

module.exports = {
  // Core
  Pipeline, load, loadSync, register,
  encodeBundle, decodeBundle, validateBundle,
  normalizeX, normalizeY,
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score, logLoss, rocAuc,
  kFold, stratifiedKFold, trainTestSplit, crossValScore,
  StandardScaler, MinMaxScaler, Preprocessor,
  // AutoML
  autoFit,
  // Ensemble
  StackingEnsemble, VotingEnsemble, BaggedEstimator,
  // Models
  LinearModel, SVMModel, XGBModel, LGBModel, KNNModel, EBMModel,
  TsetlinModel, BARTModel,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor,
  XLearnLRClassifier, XLearnLRRegressor,
}
