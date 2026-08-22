const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  Pipeline, encodeBundle, register, RegistryError, ValidationError
} = require('@wlearn/core')
const { VotingEnsemble } = require('../src/voting.js')
const { BaggedEstimator } = require('../src/bagging.js')
const { StackingEnsemble } = require('../src/stacking.js')

function childBundle(typeId) {
  return encodeBundle({ typeId }, [])
}

function nestedArtifact(id, typeId) {
  return {
    id,
    data: childBundle(typeId),
    mediaType: 'application/x-wlearn-bundle',
  }
}

async function assertTransactionalCleanup(load, outerTypeId, manifestFields, ids, prefix) {
  const firstTypeId = `wlearn.test.${prefix}.first@1`
  const failingTypeId = `wlearn.test.${prefix}.failing@1`
  let disposed = 0
  register(firstTypeId, () => ({
    classes: new Int32Array([0, 1]),
    capabilities: { predictProba: true },
    predictProba() { return new Float64Array() },
    dispose() { disposed++ },
  }))
  register(failingTypeId, () => { throw new Error(`${prefix} loader failure`) })
  const artifacts = ids.map((id, index) => nestedArtifact(
    id, index === 0 ? firstTypeId : failingTypeId
  ))
  const bytes = encodeBundle({ typeId: outerTypeId, ...manifestFields }, artifacts)
  await assert.rejects(() => load(bytes), new RegExp(`${prefix} loader failure`))
  assert.equal(disposed, 1, `${prefix} must dispose a child loaded before failure`)
}

async function assertContextForwarding(load, outerTypeId, manifestFields, ids, prefix) {
  const childTypeId = `wlearn.test.${prefix}.context@1`
  const contexts = []
  register(childTypeId, (manifest, toc, blobs, context) => {
    contexts.push(context)
    return {
      classes: new Int32Array([0, 1]),
      capabilities: { predictProba: true },
      predictProba() { return new Float64Array() },
      dispose() {},
    }
  }, { acceptsContext: true, sync: true })
  const bytes = encodeBundle({ typeId: outerTypeId, ...manifestFields }, ids.map(
    id => nestedArtifact(id, childTypeId)
  ))
  const runtimeOptions = { limits: { maxApplyRows: 2 } }
  const loaded = await load(bytes, {
    'wlearn.preprocess.tabular@1': runtimeOptions
  })
  assert.equal(contexts.length, ids.length)
  for (const context of contexts) {
    assert(Object.isFrozen(context))
    assert.deepEqual(context.loaderOptions['wlearn.preprocess.tabular@1'], runtimeOptions)
    assert.notStrictEqual(
      context.loaderOptions['wlearn.preprocess.tabular@1'], runtimeOptions
    )
  }
  assert(contexts.every(context => context === contexts[0]))
  loaded.dispose()
}

describe('direct composite load hardening', () => {
  it('Pipeline rejects wrong outer types, preflights requirements, and cleans up', async () => {
    const wrongType = encodeBundle({
      typeId: 'wlearn.test.not-pipeline@1',
      steps: [{ name: 'model', params: {} }],
    }, [])
    await assert.rejects(() => Pipeline.load(wrongType), ValidationError)

    const availableTypeId = 'wlearn.test.pipeline-preflight.available@1'
    const missingTypeId = 'wlearn.test.pipeline-preflight.missing@1'
    let dispatched = 0
    register(availableTypeId, () => {
      dispatched++
      return { dispose() {} }
    })
    const preflight = encodeBundle({
      typeId: 'wlearn.pipeline@1',
      steps: [
        { name: 'first', params: {} },
        { name: 'second', params: {} },
      ],
    }, [
      nestedArtifact('first', availableTypeId),
      nestedArtifact('second', missingTypeId),
    ])
    await assert.rejects(() => Pipeline.load(preflight), RegistryError)
    assert.equal(dispatched, 0, 'preflight must run before the first child loader')

    await assertTransactionalCleanup(
      bytes => Pipeline.load(bytes),
      'wlearn.pipeline@1',
      { steps: [
        { name: 'first', params: {} },
        { name: 'second', params: {} },
      ] },
      ['first', 'second'],
      'pipeline-transaction'
    )
  })

  it('VotingEnsemble rejects wrong outer types and cleans up partial loads', async () => {
    const params = {
      task: 'classification', voting: 'soft', weights: [0.5, 0.5],
      estimatorNames: ['first', 'second'], classes: [0, 1],
    }
    const wrongType = encodeBundle({
      typeId: 'wlearn.test.not-voting@1', params
    }, [])
    await assert.rejects(() => VotingEnsemble.load(wrongType), ValidationError)
    await assertTransactionalCleanup(
      bytes => VotingEnsemble.load(bytes),
      'wlearn.ensemble.voting.classifier@1',
      { params },
      params.estimatorNames,
      'voting-transaction'
    )
  })

  it('BaggedEstimator rejects wrong outer types and cleans up partial loads', async () => {
    const params = {
      task: 'classification', kFold: 2, nRepeats: 1, seed: 42,
      estimatorName: 'base', classes: [0, 1], nClasses: 2, nSamples: 2,
    }
    const wrongType = encodeBundle({
      typeId: 'wlearn.test.not-bagging@1', params
    }, [])
    await assert.rejects(() => BaggedEstimator.load(wrongType), ValidationError)
    await assertTransactionalCleanup(
      bytes => BaggedEstimator.load(bytes),
      'wlearn.ensemble.bagged.classifier@1',
      { params },
      ['fold_0', 'fold_1'],
      'bagging-transaction'
    )
  })

  it('keeps legacy bagging inference but rejects unavailable OOF evidence', async () => {
    const childTypeId = 'wlearn.test.legacy-bag-child@1'
    let metaCreates = 0
    register(childTypeId, () => ({
      classes: new Int32Array([0, 1]),
      capabilities: { predictProba: true },
      predictProba(input) {
        const output = new Float64Array(input.rows * 2)
        for (let row = 0; row < input.rows; row++) {
          output[row * 2] = 0.75
          output[row * 2 + 1] = 0.25
        }
        return output
      },
      save() { return childBundle(childTypeId) },
      dispose() {},
    }), { sync: true })
    const legacy = encodeBundle({
      typeId: 'wlearn.ensemble.bagged.classifier@1',
      params: {
        task: 'classification', kFold: 2, nRepeats: 1, seed: 42,
        estimatorName: 'base', classes: [0, 1], nClasses: 2, nSamples: 4,
      },
    }, [nestedArtifact('fold_0', childTypeId), nestedArtifact('fold_1', childTypeId)])
    const bag = await BaggedEstimator.load(legacy)
    const X = { data: new Float64Array([0, 1, 2, 3]), rows: 4, cols: 1 }
    const y = new Int32Array([0, 0, 1, 1])

    assert.equal(bag.isFitted, true)
    assert.deepEqual([...bag.predictProba(X)], [
      0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25,
    ])
    assert.throws(() => bag.oofPredictions, /does not include stored OOF/)
    assert.throws(() => bag.save(), /does not include stored OOF/)

    class NeverCreatedMeta {
      static async create() { metaCreates++; return new NeverCreatedMeta() }
    }
    const stacking = await StackingEnsemble.create({
      estimators: [['legacy', bag]],
      finalEstimator: ['meta', NeverCreatedMeta, {}],
      cv: 2,
      task: 'classification',
    })
    await assert.rejects(() => stacking.fit(X, y), /does not include stored OOF/)
    assert.equal(metaCreates, 0)
    assert.equal(bag.isFitted, true, 'rejected stacking fit does not take ownership')
    stacking.dispose()
    bag.dispose()
  })

  it('StackingEnsemble rejects wrong outer types and cleans up partial loads', async () => {
    const params = {
      task: 'classification', cv: 2, passthrough: false, seed: 42,
      estimatorNames: ['base'], metaName: 'meta', classes: [0, 1],
      nMetaCols: 2,
    }
    const wrongType = encodeBundle({
      typeId: 'wlearn.test.not-stacking@1', params
    }, [])
    await assert.rejects(() => StackingEnsemble.load(wrongType), ValidationError)
    await assertTransactionalCleanup(
      bytes => StackingEnsemble.load(bytes),
      'wlearn.ensemble.stacking.classifier@1',
      { params },
      ['base', 'meta'],
      'stacking-transaction'
    )
  })

  it('voting, bagging, and stacking forward one immutable recursive context', async () => {
    await assertContextForwarding(
      (bytes, loaderOptions) => VotingEnsemble.load(bytes, { loaderOptions }),
      'wlearn.ensemble.voting.classifier@1',
      { params: {
        task: 'classification', voting: 'soft', weights: [0.5, 0.5],
        estimatorNames: ['first', 'second'], classes: [0, 1]
      } },
      ['first', 'second'],
      'voting-context'
    )
    await assertContextForwarding(
      (bytes, loaderOptions) => BaggedEstimator.load(bytes, { loaderOptions }),
      'wlearn.ensemble.bagged.classifier@1',
      { params: {
        task: 'classification', kFold: 2, nRepeats: 1, seed: 42,
        estimatorName: 'base', classes: [0, 1], nClasses: 2, nSamples: 2
      } },
      ['fold_0', 'fold_1'],
      'bagging-context'
    )
    await assertContextForwarding(
      (bytes, loaderOptions) => StackingEnsemble.load(bytes, { loaderOptions }),
      'wlearn.ensemble.stacking.classifier@1',
      { params: {
        task: 'classification', cv: 2, passthrough: false, seed: 42,
        estimatorNames: ['base'], metaName: 'meta', classes: [0, 1],
        nMetaCols: 2
      } },
      ['base', 'meta'],
      'stacking-context'
    )
  })

  it('rejects hash-valid but semantically inconsistent ensemble manifests', async () => {
    const childTypeId = 'wlearn.test.ensemble-semantic-child@1'
    let dispatched = 0
    register(childTypeId, () => {
      dispatched++
      return { dispose() {} }
    })
    const child = id => nestedArtifact(id, childTypeId)

    const voting = encodeBundle({
      typeId: 'wlearn.ensemble.voting.classifier@1',
      params: {
        task: 'classification', voting: 'soft', weights: [1],
        estimatorNames: ['first', 'second'], classes: [0, 1]
      }
    }, [child('first'), child('second')])
    await assert.rejects(
      () => VotingEnsemble.load(voting),
      /weights.*matching estimatorNames/
    )

    const stacking = encodeBundle({
      typeId: 'wlearn.ensemble.stacking.classifier@1',
      params: {
        task: 'classification', cv: 2, passthrough: false, seed: 42,
        estimatorNames: ['base'], metaName: 'base', classes: [0, 1],
        nMetaCols: 2
      }
    }, [child('base')])
    await assert.rejects(
      () => StackingEnsemble.load(stacking), /metaName must differ/
    )

    const bagging = encodeBundle({
      typeId: 'wlearn.ensemble.bagged.classifier@1',
      params: {
        task: 'classification', kFold: 2, nRepeats: 1, seed: 42,
        estimatorName: 'base', classes: [0, 1], nClasses: 2, nSamples: 2
      }
    }, [
      child('fold_0'), child('fold_1'),
      { id: 'oof', data: new Uint8Array(8), mediaType: 'application/octet-stream' }
    ])
    await assert.rejects(
      () => BaggedEstimator.load(bagging), /OOF artifact length/
    )

    const oversizedBagging = encodeBundle({
      typeId: 'wlearn.ensemble.bagged.classifier@1',
      params: {
        task: 'classification', kFold: 100000000, nRepeats: 1, seed: 42,
        estimatorName: 'base', classes: [0, 1], nClasses: 2, nSamples: 2
      }
    }, [])
    await assert.rejects(
      () => BaggedEstimator.load(oversizedBagging), /artifact count/
    )
    assert.equal(dispatched, 0, 'semantic preflight must run before child loaders')
  })
})
