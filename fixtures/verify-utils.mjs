import assert from 'node:assert/strict'

export function assertFiniteClose(actual, expected, tolerance, message) {
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    throw new Error(`${message}: predictions must be finite, got actual=${actual}, expected=${expected}`)
  }
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${message}: expected ~${expected}, got ${actual} (tol=${tolerance})`)
  }
}

export function assertPredictionParity(actual, expected, tolerance, message = 'predictions') {
  assert.equal(actual.length, expected.length, `${message}: length differs`)
  assert.ok(actual.length > 0, `${message}: predictions must not be empty`)
  for (let i = 0; i < actual.length; i++) {
    assertFiniteClose(actual[i], expected[i], tolerance, `${message}[${i}]`)
  }
}
