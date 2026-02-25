export class WlearnError extends Error {
  constructor(message, code) {
    super(message)
    this.name = 'WlearnError'
    this.code = code || 'ERR_WLEARN'
  }
}

export class BundleError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_BUNDLE')
    this.name = 'BundleError'
  }
}

export class RegistryError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_REGISTRY')
    this.name = 'RegistryError'
  }
}

export class ValidationError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_VALIDATION')
    this.name = 'ValidationError'
  }
}

export class NotFittedError extends WlearnError {
  constructor(message = 'Model is not fitted. Call fit() first.') {
    super(message, 'ERR_NOT_FITTED')
    this.name = 'NotFittedError'
  }
}

export class DisposedError extends WlearnError {
  constructor(message = 'Model has been disposed.') {
    super(message, 'ERR_DISPOSED')
    this.name = 'DisposedError'
  }
}
