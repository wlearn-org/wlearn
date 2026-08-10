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
  register(firstTypeId, () => ({ dispose() { disposed++ } }))
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
    return { dispose() {} }
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
    assert.strictEqual(
      context.loaderOptions['wlearn.preprocess.tabular@1'], runtimeOptions
    )
  }
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
})
