const { VotingEnsemble } = require('./voting.js')
const { StackingEnsemble } = require('./stacking.js')
const { BaggedEstimator } = require('./bagging.js')
const { caruanaSelect } = require('./selection.js')
const { getOofPredictions } = require('./oof.js')
const { optimizeWeights, projectSimplex } = require('./weights.js')

module.exports = {
  VotingEnsemble,
  StackingEnsemble,
  BaggedEstimator,
  caruanaSelect,
  getOofPredictions,
  optimizeWeights,
  projectSimplex,
}
