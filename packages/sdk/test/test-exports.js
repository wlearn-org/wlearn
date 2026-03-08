const { test } = require('node:test')
const assert = require('node:assert/strict')
const sdk = require('@wlearn/sdk')

test('core exports', () => {
  assert.equal(typeof sdk.Pipeline, 'function')
  assert.equal(typeof sdk.load, 'function')
  assert.equal(typeof sdk.loadSync, 'function')
  assert.equal(typeof sdk.register, 'function')
  assert.equal(typeof sdk.encodeBundle, 'function')
  assert.equal(typeof sdk.decodeBundle, 'function')
  assert.equal(typeof sdk.validateBundle, 'function')
  assert.equal(typeof sdk.normalizeX, 'function')
  assert.equal(typeof sdk.normalizeY, 'function')
})

test('metric exports', () => {
  assert.equal(typeof sdk.accuracy, 'function')
  assert.equal(typeof sdk.r2Score, 'function')
  assert.equal(typeof sdk.meanSquaredError, 'function')
  assert.equal(typeof sdk.meanAbsoluteError, 'function')
  assert.equal(typeof sdk.confusionMatrix, 'function')
  assert.equal(typeof sdk.precisionScore, 'function')
  assert.equal(typeof sdk.recallScore, 'function')
  assert.equal(typeof sdk.f1Score, 'function')
  assert.equal(typeof sdk.logLoss, 'function')
  assert.equal(typeof sdk.rocAuc, 'function')
})

test('cv exports', () => {
  assert.equal(typeof sdk.kFold, 'function')
  assert.equal(typeof sdk.stratifiedKFold, 'function')
  assert.equal(typeof sdk.trainTestSplit, 'function')
  assert.equal(typeof sdk.crossValScore, 'function')
})

test('preprocessing exports', () => {
  assert.equal(typeof sdk.StandardScaler, 'function')
  assert.equal(typeof sdk.MinMaxScaler, 'function')
  assert.equal(typeof sdk.Preprocessor, 'function')
})

test('automl export', () => {
  assert.equal(typeof sdk.autoFit, 'function')
})

test('ensemble exports', () => {
  assert.equal(typeof sdk.StackingEnsemble, 'function')
  assert.equal(typeof sdk.VotingEnsemble, 'function')
  assert.equal(typeof sdk.BaggedEstimator, 'function')
})

test('model exports', () => {
  assert.equal(typeof sdk.LinearModel, 'function')
  assert.equal(typeof sdk.SVMModel, 'function')
  assert.equal(typeof sdk.XGBModel, 'function')
  assert.equal(typeof sdk.LGBModel, 'function')
  assert.equal(typeof sdk.KNNModel, 'function')
  assert.equal(typeof sdk.EBMModel, 'function')
  assert.equal(typeof sdk.TsetlinModel, 'function')
  assert.equal(typeof sdk.BARTModel, 'function')
  assert.equal(typeof sdk.XLearnFMClassifier, 'function')
  assert.equal(typeof sdk.XLearnFMRegressor, 'function')
  assert.equal(typeof sdk.XLearnFFMClassifier, 'function')
  assert.equal(typeof sdk.XLearnFFMRegressor, 'function')
  assert.equal(typeof sdk.XLearnLRClassifier, 'function')
  assert.equal(typeof sdk.XLearnLRRegressor, 'function')
})

test('no mitra (requires onnxruntime peer dep)', () => {
  assert.equal(sdk.MitraClassifier, undefined)
  assert.equal(sdk.MitraRegressor, undefined)
})
