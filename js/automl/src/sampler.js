const { makeLCG } = require('@wlearn/core')
const { conditionOrder, conditionSatisfied } = require('./conditions.js')

const { floor, round, log, exp, min, max } = Math

/**
 * Sample a single value from a SearchParam definition.
 */
function sampleParam(param, rng) {
  const { type } = param
  switch (type) {
    case 'categorical':
      return param.values[floor(rng() * param.values.length)]
    case 'uniform':
      return param.low + rng() * (param.high - param.low)
    case 'log_uniform':
      return exp(log(param.low) + rng() * (log(param.high) - log(param.low)))
    case 'int_uniform':
      return param.low + floor(rng() * (param.high - param.low + 1))
    case 'int_log_uniform':
      return round(exp(log(param.low) + rng() * (log(param.high) - log(param.low))))
    default:
      throw new Error(`Unknown SearchParam type: "${type}"`)
  }
}

/**
 * Sample a complete config from a SearchSpace, respecting conditions.
 */
function sampleConfig(space, rng) {
  const config = {}
  for (const key of conditionOrder(space)) {
    if (conditionSatisfied(space[key].condition, config)) {
      config[key] = sampleParam(space[key], rng)
    }
  }

  return config
}

/**
 * Generate n random configs from a SearchSpace.
 */
function randomConfigs(space, n, { seed = 42 } = {}) {
  const rng = makeLCG(seed)
  const configs = []
  for (let i = 0; i < n; i++) {
    configs.push(sampleConfig(space, rng))
  }
  return configs
}

/**
 * Enumerate grid points from a SearchSpace.
 * Continuous params discretized to `steps` values.
 */
function gridConfigs(space, { steps = 5 } = {}) {
  let combos = [{}]
  for (const key of conditionOrder(space)) {
    const vals = _discretize(space[key], steps)
    combos = combos.flatMap(combo => conditionSatisfied(space[key].condition, combo)
      ? vals.map(value => ({ ...combo, [key]: value }))
      : [combo])
  }

  return combos
}

function _discretize(param, steps) {
  const { type } = param
  switch (type) {
    case 'categorical':
      return [...param.values]
    case 'uniform': {
      const arr = []
      for (let i = 0; i < steps; i++) {
        arr.push(param.low + (param.high - param.low) * i / max(1, steps - 1))
      }
      return arr
    }
    case 'log_uniform': {
      const logLow = log(param.low)
      const logHigh = log(param.high)
      const arr = []
      for (let i = 0; i < steps; i++) {
        arr.push(exp(logLow + (logHigh - logLow) * i / max(1, steps - 1)))
      }
      return arr
    }
    case 'int_uniform': {
      const range = param.high - param.low + 1
      if (range <= steps) {
        const arr = []
        for (let v = param.low; v <= param.high; v++) arr.push(v)
        return arr
      }
      const arr = []
      for (let i = 0; i < steps; i++) {
        arr.push(param.low + round((param.high - param.low) * i / max(1, steps - 1)))
      }
      return [...new Set(arr)].sort((a, b) => a - b)
    }
    case 'int_log_uniform': {
      const logLow = log(param.low)
      const logHigh = log(param.high)
      const arr = []
      for (let i = 0; i < steps; i++) {
        arr.push(round(exp(logLow + (logHigh - logLow) * i / max(1, steps - 1))))
      }
      return [...new Set(arr)].sort((a, b) => a - b)
    }
    default:
      throw new Error(`Unknown SearchParam type: "${type}"`)
  }
}

module.exports = { sampleParam, sampleConfig, randomConfigs, gridConfigs }
