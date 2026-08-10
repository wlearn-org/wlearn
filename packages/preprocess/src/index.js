'use strict'

const { createPreprocessAPI } = require('./factory.js')

module.exports = createPreprocessAPI(() => require('tranfi'), 'native')
