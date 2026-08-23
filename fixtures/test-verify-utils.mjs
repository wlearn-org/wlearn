import assert from 'node:assert/strict'
import { assertFiniteClose, assertPredictionParity } from './verify-utils.mjs'

assert.doesNotThrow(() => assertFiniteClose(1.000001, 1, 1e-5, 'finite'))
assert.throws(() => assertFiniteClose(Number.NaN, 1, 1e-5, 'nan'), /must be finite/)
assert.throws(() => assertFiniteClose(1, Number.NaN, 1e-5, 'nan'), /must be finite/)
assert.throws(() => assertFiniteClose(Number.POSITIVE_INFINITY, 1, 1e-5, 'infinity'), /must be finite/)
assert.throws(() => assertPredictionParity([], [1], 1e-5), /length differs/)
assert.throws(() => assertPredictionParity([1], [], 1e-5), /length differs/)
assert.throws(() => assertPredictionParity([], [], 1e-5), /must not be empty/)
assert.throws(() => assertPredictionParity([Number.NaN], [Number.NaN], 1e-5), /must be finite/)
assert.doesNotThrow(() => assertPredictionParity([1, 2], [1, 2.000001], 1e-5))

console.log('Fixture verifier guard probes passed')
