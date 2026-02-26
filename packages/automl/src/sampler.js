import { makeLCG } from '@wlearn/core'

const { floor, round, log, exp, min, max } = Math

/**
 * Sample a single value from a SearchParam definition.
 */
export function sampleParam(param, rng) {
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
export function sampleConfig(space, rng) {
  const config = {}
  const keys = Object.keys(space)

  // First pass: non-conditional params
  for (const key of keys) {
    if (!space[key].condition) {
      config[key] = sampleParam(space[key], rng)
    }
  }

  // Second pass: conditional params
  for (const key of keys) {
    const { condition } = space[key]
    if (!condition) continue
    let satisfied = true
    for (const [ck, cv] of Object.entries(condition)) {
      if (config[ck] !== cv) { satisfied = false; break }
    }
    if (satisfied) {
      config[key] = sampleParam(space[key], rng)
    }
  }

  return config
}

/**
 * Generate n random configs from a SearchSpace.
 */
export function randomConfigs(space, n, { seed = 42 } = {}) {
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
export function gridConfigs(space, { steps = 5 } = {}) {
  const keys = Object.keys(space)
  if (keys.length === 0) return [{}]

  // Build value arrays for non-conditional params
  const nonCond = keys.filter(k => !space[k].condition)
  const condKeys = keys.filter(k => space[k].condition)

  const valueArrays = nonCond.map(k => _discretize(space[k], steps))

  // Cartesian product of non-conditional params
  let combos = [{}]
  for (let i = 0; i < nonCond.length; i++) {
    const key = nonCond[i]
    const vals = valueArrays[i]
    const next = []
    for (const combo of combos) {
      for (const v of vals) {
        next.push({ ...combo, [key]: v })
      }
    }
    combos = next
  }

  // Add conditional params where conditions are met
  for (const combo of combos) {
    for (const key of condKeys) {
      const { condition } = space[key]
      let satisfied = true
      for (const [ck, cv] of Object.entries(condition)) {
        if (combo[ck] !== cv) { satisfied = false; break }
      }
      if (satisfied) {
        // For grid, take all discrete values and expand
        // But that would multiply combos -- for simplicity, take midpoint
        const vals = _discretize(space[key], steps)
        combo[key] = vals[floor(vals.length / 2)]
      }
    }
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
