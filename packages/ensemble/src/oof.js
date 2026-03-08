const { stratifiedKFold, kFold, normalizeX, normalizeY, ValidationError } = require('@wlearn/core')

/**
 * Generate out-of-fold predictions for a list of estimator specs.
 */
async function getOofPredictions(estimatorSpecs, X, y, {
  cv = 5,
  seed = 42,
  task = 'classification',
} = {}) {
  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const n = Xn.rows

  const folds = task === 'classification'
    ? stratifiedKFold(yn, cv, { shuffle: true, seed })
    : kFold(n, cv, { shuffle: true, seed })

  // Discover classes for classification
  let classes = null
  let nClasses = 0
  if (task === 'classification') {
    const labelSet = new Set()
    for (let i = 0; i < yn.length; i++) labelSet.add(yn[i])
    classes = new Int32Array([...labelSet].sort((a, b) => a - b))
    nClasses = classes.length
  }

  const oofPreds = []

  for (const [name, EstimatorClass, params] of estimatorSpecs) {
    let oof
    if (task === 'classification') {
      oof = new Float64Array(n * nClasses)
    } else {
      oof = new Float64Array(n)
    }

    for (const { train, test } of folds) {
      const Xtrain = _subsetX(Xn, train)
      const ytrain = _subsetY(yn, train)
      const Xtest = _subsetX(Xn, test)

      const model = await EstimatorClass.create(params || {})
      try {
        model.fit(Xtrain, ytrain)
        if (task === 'classification') {
          const proba = await model.predictProba(Xtest)
          for (let i = 0; i < test.length; i++) {
            const row = test[i]
            for (let c = 0; c < nClasses; c++) {
              oof[row * nClasses + c] = proba[i * nClasses + c]
            }
          }
        } else {
          const preds = await model.predict(Xtest)
          for (let i = 0; i < test.length; i++) {
            oof[test[i]] = preds[i]
          }
        }
      } finally {
        model.dispose()
      }
    }
    oofPreds.push(oof)
  }

  return { oofPreds, classes }
}

// --- Internal helpers (same as core/cv.js) ---

function _subsetX(X, indices) {
  const { data, cols } = X
  const rows = indices.length
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const srcOff = indices[i] * cols
    out.set(data.subarray(srcOff, srcOff + cols), i * cols)
  }
  return { data: out, rows, cols }
}

function _subsetY(y, indices) {
  const out = new (y.constructor)(indices.length)
  for (let i = 0; i < indices.length; i++) {
    out[i] = y[indices[i]]
  }
  return out
}

module.exports = { getOofPredictions }
