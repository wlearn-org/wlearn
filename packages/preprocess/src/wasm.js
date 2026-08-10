'use strict'

const { createPreprocessAPI } = require('./factory.js')

module.exports = createPreprocessAPI(async () => {
  const createTranfi = require('tranfi/wasm')
  return createTranfi()
}, 'wasm')
