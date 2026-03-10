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

// Mitra requires onnxruntime peer dep -- optional
let MitraModel, MitraClassifier, MitraRegressor, registerMitraLoaders
try {
  const mitra = require('@wlearn/mitra')
  MitraModel = mitra.MitraModel
  MitraClassifier = mitra.MitraClassifier
  MitraRegressor = mitra.MitraRegressor
  registerMitraLoaders = mitra.registerLoaders
} catch (_) {}

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
  // Mitra (optional)
  MitraModel, MitraClassifier, MitraRegressor, registerMitraLoaders,
}
