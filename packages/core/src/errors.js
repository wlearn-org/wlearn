class WlearnError extends Error {
  constructor(message, code) {
    super(message)
    this.name = 'WlearnError'
    this.code = code || 'ERR_WLEARN'
  }
}

class BundleError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_BUNDLE')
    this.name = 'BundleError'
  }
}

class RegistryError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_REGISTRY')
    this.name = 'RegistryError'
  }
}

class ValidationError extends WlearnError {
  constructor(message) {
    super(message, 'ERR_VALIDATION')
    this.name = 'ValidationError'
  }
}

class NotFittedError extends WlearnError {
  constructor(message = 'Model is not fitted. Call fit() first.') {
    super(message, 'ERR_NOT_FITTED')
    this.name = 'NotFittedError'
  }
}

class DisposedError extends WlearnError {
  constructor(message = 'Model has been disposed.') {
    super(message, 'ERR_DISPOSED')
    this.name = 'DisposedError'
  }
}

class ResourceLimitError extends WlearnError {
  constructor(message = 'Resource limit exceeded.') {
    super(message, 'ERR_RESOURCE_LIMIT')
    this.name = 'ResourceLimitError'
  }
}

class CancelledError extends WlearnError {
  constructor(message = 'Operation cancelled.') {
    super(message, 'ERR_CANCELLED')
    this.name = 'CancelledError'
  }
}

class BackendError extends WlearnError {
  constructor(message = 'Backend operation failed.') {
    super(message, 'ERR_BACKEND')
    this.name = 'BackendError'
  }
}

module.exports = {
  WlearnError,
  BundleError,
  RegistryError,
  ValidationError,
  NotFittedError,
  DisposedError,
  ResourceLimitError,
  CancelledError,
  BackendError
}
