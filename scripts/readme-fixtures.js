// User-supplied data for examples; API imports and fitted objects belong to the
// explicit case context. Training, calibration and test rows are separate.
{
  const rows = (start, n) => Array.from({ length: n }, (_, i) => {
    const j = start + i
    return [j % 2 ? 1 + j / 200 : -1 - j / 200, (j % 7) / 7, (j % 11) / 11, (j % 5) / 5]
  })
  const X = rows(0, 80), Xtest = rows(160, 8), xc = rows(80, 80)
  const y = X.map(r => +(r[0] > 0)), yt = Xtest.map(r => +(r[0] > 0)), yc = xc.map(r => +(r[0] > 0))
  Object.assign(globalThis, {
    X, y, X_train: X, y_train: y, XTrain: X, yTrain: y,
    X_test: Xtest, Xtest, X_new: Xtest, XTest: Xtest, yTest: yt, y_test: yt,
    XCalibration: xc, yCalibration: yc, calibrationX: xc, calibrationY: yc,
    testX: Xtest, testY: yt, calibrationPredictions: yc.map(v => v + .1),
    calibrationTargets: yc, testPredictions: yt.map(v => v + .1), testTargets: yt,
    calibrationProbabilities: yc.map(v => v ? [.1, .9] : [.8, .2]),
    calibrationLabels: yc.map(v => [1-v, v]), testProbabilities: [[.3, .7], [.8, .2]],
    calibrationConfidence: yc.map(() => .9), calibrationErrors: yc.map(() => 0), testConfidence: [.4, .9],
    X_imbalanced: X, y_imbalanced: y, X_large: X, y_large: y,
    labels: y, trueLabels: y, predLabels: y, yTrue: yt, yPred: yt,
    normalData: X, dummyLabels: y.map(() => 1), testData: Xtest,
    time: X.map((_, i) => i + 1), status: y,
    nClasses: 2, nTasks: 2, Y: y.flatMap(v => [v, 2*v]),
    evaluate: p => -Object.values(p).reduce((s,v) => s + (typeof v === 'number' ? v*v : 0), 0)
  })
}
