/**
 * Promise-lifting utilities for MaybePromise<T> support.
 *
 * Allows pipeline/ensemble to handle both sync (WASM) and async (ONNX)
 * estimators without forcing async on sync callers.
 */

/**
 * Check if a value is thenable (Promise-like).
 * @param {*} x
 * @returns {boolean}
 */
export function isPromiseLike(x) {
  return x != null && typeof x.then === 'function'
}

/**
 * Apply f to x, propagating Promise only if x is thenable.
 * Returns T when x is T, Promise<T> when x is Promise<T>.
 *
 * @template T, U
 * @param {T | Promise<T>} x
 * @param {(value: T) => U} f
 * @returns {U | Promise<U>}
 */
export function lift(x, f) {
  return isPromiseLike(x) ? x.then(f) : f(x)
}
