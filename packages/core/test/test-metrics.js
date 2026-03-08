const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc
} = require('../src/metrics.js')
const { ValidationError } = require('../src/errors.js')

function approx(actual, expected, tol = 1e-9) {
  assert(Math.abs(actual - expected) < tol,
    `expected ~${expected}, got ${actual} (diff=${Math.abs(actual - expected)})`)
}

describe('accuracy', () => {
  it('perfect predictions', () => {
    const y = new Int32Array([0, 1, 2, 0, 1])
    approx(accuracy(y, y), 1.0)
  })

  it('all wrong', () => {
    approx(accuracy(
      new Int32Array([0, 0, 0]),
      new Int32Array([1, 1, 1])
    ), 0)
  })

  it('partial', () => {
    approx(accuracy(
      new Int32Array([0, 1, 0, 1]),
      new Int32Array([0, 1, 1, 0])
    ), 0.5)
  })

  it('throws on length mismatch', () => {
    assert.throws(
      () => accuracy(new Int32Array([0, 1]), new Int32Array([0])),
      ValidationError
    )
  })

  it('throws on empty', () => {
    assert.throws(() => accuracy(new Int32Array([]), new Int32Array([])), ValidationError)
  })
})

describe('r2Score', () => {
  it('perfect predictions', () => {
    const y = new Float64Array([1, 2, 3, 4, 5])
    approx(r2Score(y, y), 1.0)
  })

  it('mean predictions give 0', () => {
    const y = new Float64Array([1, 2, 3, 4, 5])
    const mean = new Float64Array([3, 3, 3, 3, 3])
    approx(r2Score(y, mean), 0)
  })

  it('worse than mean gives negative', () => {
    const y = new Float64Array([1, 2, 3])
    const bad = new Float64Array([10, 10, 10])
    assert(r2Score(y, bad) < 0)
  })

  it('constant true returns 0', () => {
    const y = new Float64Array([5, 5, 5])
    const pred = new Float64Array([4, 5, 6])
    approx(r2Score(y, pred), 0)
  })
})

describe('meanSquaredError', () => {
  it('zero error', () => {
    const y = new Float64Array([1, 2, 3])
    approx(meanSquaredError(y, y), 0)
  })

  it('known value', () => {
    approx(meanSquaredError(
      new Float64Array([1, 2, 3]),
      new Float64Array([2, 3, 4])
    ), 1.0)
  })
})

describe('meanAbsoluteError', () => {
  it('zero error', () => {
    const y = new Float64Array([1, 2, 3])
    approx(meanAbsoluteError(y, y), 0)
  })

  it('known value', () => {
    approx(meanAbsoluteError(
      new Float64Array([1, 2, 3]),
      new Float64Array([2, 1, 5])
    ), 4 / 3)
  })
})

describe('confusionMatrix', () => {
  it('binary', () => {
    const yTrue = new Int32Array([0, 0, 1, 1])
    const yPred = new Int32Array([0, 1, 0, 1])
    const { matrix, labels } = confusionMatrix(yTrue, yPred)
    assert.deepEqual([...labels], [0, 1])
    // [[1,1],[1,1]] row=true, col=pred
    assert.deepEqual([...matrix], [1, 1, 1, 1])
  })

  it('multiclass', () => {
    const yTrue = new Int32Array([0, 1, 2, 0, 1, 2])
    const yPred = new Int32Array([0, 2, 1, 0, 1, 2])
    const { matrix, labels } = confusionMatrix(yTrue, yPred)
    assert.deepEqual([...labels], [0, 1, 2])
    // Class 0: 2 correct
    // Class 1: 1 correct, 1 predicted as 2
    // Class 2: 1 correct, 1 predicted as 1
    assert.equal(matrix[0 * 3 + 0], 2) // true=0, pred=0
    assert.equal(matrix[1 * 3 + 1], 1) // true=1, pred=1
    assert.equal(matrix[1 * 3 + 2], 1) // true=1, pred=2
    assert.equal(matrix[2 * 3 + 1], 1) // true=2, pred=1
    assert.equal(matrix[2 * 3 + 2], 1) // true=2, pred=2
  })
})

describe('precisionScore', () => {
  it('binary perfect', () => {
    approx(precisionScore(
      new Int32Array([0, 0, 1, 1]),
      new Int32Array([0, 0, 1, 1])
    ), 1.0)
  })

  it('binary known', () => {
    // true=[0,0,1,1], pred=[0,1,0,1]
    // For positive class (1): TP=1, FP=1 -> precision=0.5
    approx(precisionScore(
      new Int32Array([0, 0, 1, 1]),
      new Int32Array([0, 1, 0, 1])
    ), 0.5)
  })

  it('micro averaging', () => {
    // 3-class: micro precision = total TP / (total TP + total FP)
    const yTrue = new Int32Array([0, 1, 2, 0, 1, 2])
    const yPred = new Int32Array([0, 2, 1, 0, 1, 2])
    // TP: class0=2, class1=1, class2=1 = 4
    // FP: class0=0, class1=1, class2=1 = 2
    approx(precisionScore(yTrue, yPred, { average: 'micro' }), 4 / 6)
  })

  it('macro averaging', () => {
    const yTrue = new Int32Array([0, 1, 2, 0, 1, 2])
    const yPred = new Int32Array([0, 2, 1, 0, 1, 2])
    // class0: TP=2, FP=0 -> p=1.0
    // class1: TP=1, FP=1 -> p=0.5
    // class2: TP=1, FP=1 -> p=0.5
    approx(precisionScore(yTrue, yPred, { average: 'macro' }), (1 + 0.5 + 0.5) / 3)
  })

  it('throws binary with >2 classes', () => {
    assert.throws(
      () => precisionScore(
        new Int32Array([0, 1, 2]),
        new Int32Array([0, 1, 2]),
        { average: 'binary' }
      ),
      ValidationError
    )
  })
})

describe('recallScore', () => {
  it('binary known', () => {
    // true=[0,0,1,1], pred=[0,1,0,1]
    // For positive class (1): TP=1, FN=1 -> recall=0.5
    approx(recallScore(
      new Int32Array([0, 0, 1, 1]),
      new Int32Array([0, 1, 0, 1])
    ), 0.5)
  })

  it('micro equals accuracy for single-label', () => {
    const yTrue = new Int32Array([0, 1, 2, 0, 1, 2])
    const yPred = new Int32Array([0, 2, 1, 0, 1, 2])
    // micro recall = total TP / (total TP + total FN) = accuracy
    approx(
      recallScore(yTrue, yPred, { average: 'micro' }),
      accuracy(yTrue, yPred)
    )
  })
})

describe('f1Score', () => {
  it('binary perfect', () => {
    approx(f1Score(
      new Int32Array([0, 1, 0, 1]),
      new Int32Array([0, 1, 0, 1])
    ), 1.0)
  })

  it('binary known', () => {
    // precision=0.5, recall=0.5 -> F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    approx(f1Score(
      new Int32Array([0, 0, 1, 1]),
      new Int32Array([0, 1, 0, 1])
    ), 0.5)
  })

  it('all wrong returns 0', () => {
    approx(f1Score(
      new Int32Array([0, 0, 0]),
      new Int32Array([1, 1, 1])
    ), 0)
  })

  it('macro averaging', () => {
    const yTrue = new Int32Array([0, 1, 2, 0, 1, 2])
    const yPred = new Int32Array([0, 2, 1, 0, 1, 2])
    const f1macro = f1Score(yTrue, yPred, { average: 'macro' })
    // class0: p=1, r=1, f1=1
    // class1: p=0.5, r=0.5, f1=0.5
    // class2: p=0.5, r=0.5, f1=0.5
    approx(f1macro, (1 + 0.5 + 0.5) / 3)
  })
})

describe('logLoss', () => {
  it('perfect binary', () => {
    const yTrue = new Int32Array([0, 1])
    const yProba = new Float64Array([0.999, 0.001, 0.001, 0.999]) // row-major [P(0), P(1)]
    const ll = logLoss(yTrue, yProba)
    assert(ll < 0.01, `expected near-zero loss, got ${ll}`)
  })

  it('uniform proba', () => {
    const yTrue = new Int32Array([0, 1])
    const yProba = new Float64Array([0.5, 0.5, 0.5, 0.5])
    approx(logLoss(yTrue, yProba), Math.log(2))
  })

  it('clips extreme values', () => {
    const yTrue = new Int32Array([0])
    const yProba = new Float64Array([0, 1]) // P(class0)=0, would be -log(0)=Inf
    const ll = logLoss(yTrue, yProba, { nClasses: 2 })
    assert(isFinite(ll), `expected finite, got ${ll}`)
  })

  it('throws on length mismatch', () => {
    assert.throws(
      () => logLoss(new Int32Array([0, 1]), new Float64Array([0.5, 0.5])),
      ValidationError
    )
  })
})

describe('rocAuc', () => {
  it('perfect separator', () => {
    const yTrue = new Int32Array([0, 0, 1, 1])
    const scores = new Float64Array([0.1, 0.2, 0.8, 0.9])
    approx(rocAuc(yTrue, scores), 1.0)
  })

  it('random classifier ~0.5', () => {
    // Interleaved scores that don't separate classes
    const yTrue = new Int32Array([0, 1, 0, 1, 0, 1])
    const scores = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    const auc = rocAuc(yTrue, scores)
    approx(auc, 0.5, 0.2) // roughly 0.5 for non-separating scores
  })

  it('inverse separator = 0', () => {
    const yTrue = new Int32Array([0, 0, 1, 1])
    const scores = new Float64Array([0.9, 0.8, 0.1, 0.2])
    approx(rocAuc(yTrue, scores), 0)
  })

  it('throws on >2 classes', () => {
    assert.throws(
      () => rocAuc(new Int32Array([0, 1, 2]), new Float64Array([0.1, 0.5, 0.9])),
      ValidationError
    )
  })

  it('throws when only one class present', () => {
    assert.throws(
      () => rocAuc(new Int32Array([1, 1, 1]), new Float64Array([0.1, 0.5, 0.9])),
      ValidationError
    )
  })

  it('throws on length mismatch', () => {
    assert.throws(
      () => rocAuc(new Int32Array([0, 1]), new Float64Array([0.5])),
      ValidationError
    )
  })
})
