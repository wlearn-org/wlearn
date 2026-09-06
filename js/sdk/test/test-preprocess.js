const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const core = require('@wlearn/core')
const preprocess = require('@wlearn/preprocess')
const basis = require('@wlearn/basis')

// Exercise barrel wiring independently of optional/heavy model runtimes.
test('SDK Preprocessor is the persistable Tranfi adapter', async () => {
  const module = { exports: {} }
  vm.runInNewContext(fs.readFileSync(require.resolve('..'), 'utf8'), {
    module,
    require(name) {
      if (name === '@wlearn/core') return core
      if (name === '@wlearn/preprocess') return preprocess
      if (name === '@wlearn/basis') return basis
      return {}
    }
  })
  assert.equal(module.exports.Preprocessor, preprocess.Preprocessor)
  for (const name of ['BasisClassifier', 'BasisRegressor', 'BasisTransformer', 'loadBasis']) {
    assert.equal(module.exports[name], basis[name])
  }
  const prep = await module.exports.Preprocessor.create({ encode: false })
  let restored
  try {
    const X = [[0], [1], [2]]
    prep.fit(X)
    restored = await core.load(prep.save())
    assert.deepEqual(restored.transform(X), prep.transform(X))
  } finally { prep.dispose(); restored?.dispose() }
})
