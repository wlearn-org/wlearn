// Portable SearchSpace values: object key order is irrelevant, booleans are
// distinct from numbers, and an absent parent never satisfies a null condition.
function valueEqual(a, b) {
  if (a === b) return true
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every(key =>
    Object.hasOwn(b, key) && valueEqual(a[key], b[key]))
}

function conditionSatisfied(condition, config) {
  return Object.entries(condition || {}).every(([key, value]) =>
    Object.hasOwn(config, key) && valueEqual(config[key], value))
}

function conditionOrder(space) {
  const keys = Object.keys(space)
  const parents = new Map(keys.map(key => {
    const condition = space[key].condition
    if (condition != null && (typeof condition !== 'object' || Array.isArray(condition))) {
      throw new Error(`Condition for "${key}" must be an object`)
    }
    const names = Object.keys(condition || {})
    for (const name of names) {
      if (!Object.hasOwn(space, name)) throw new Error(`Unknown condition parent "${name}" for "${key}"`)
    }
    return [key, names]
  }))
  const done = new Set()
  // Stable layers retain declaration order among independent parameters and
  // preserve the RNG sequence of existing unconditional spaces.
  while (done.size < keys.length) {
    const ready = keys.filter(key => !done.has(key) && parents.get(key).every(name => done.has(name)))
    if (!ready.length) throw new Error('Cyclic SearchSpace conditions')
    for (const key of ready) done.add(key)
  }
  return [...done]
}

function effectiveSearchSpace(space, fixed = {}) {
  const context = { ...space }
  for (const [key, value] of Object.entries(fixed)) context[key] = { type: 'categorical', values: [value] }
  const effective = {}, inactive = new Set()
  for (const key of conditionOrder(context)) {
    if (Object.hasOwn(fixed, key)) continue
    const condition = {}
    for (const [parent, value] of Object.entries(context[key].condition || {})) {
      if (inactive.has(parent) || (Object.hasOwn(fixed, parent) && !valueEqual(fixed[parent], value))) {
        inactive.add(key)
        break
      }
      if (!Object.hasOwn(fixed, parent)) condition[parent] = value
    }
    if (!inactive.has(key)) effective[key] = { ...context[key], condition }
  }
  return effective
}

module.exports = { valueEqual, conditionSatisfied, conditionOrder, effectiveSearchSpace }
