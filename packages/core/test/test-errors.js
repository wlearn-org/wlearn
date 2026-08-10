'use strict'

const assert = require('node:assert/strict')
const { test } = require('node:test')

const {
  BackendError,
  CancelledError,
  ResourceLimitError
} = require('../src/index.js')

test('backend-facing error families have stable public codes', () => {
  assert.deepEqual([
    new ResourceLimitError().code,
    new CancelledError().code,
    new BackendError().code
  ], [
    'ERR_RESOURCE_LIMIT',
    'ERR_CANCELLED',
    'ERR_BACKEND'
  ])
})
