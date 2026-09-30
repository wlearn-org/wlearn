'use strict'

const tranfi = require('tranfi')

// Choose the whole API so backend identity, cancellation and loader registration
// stay consistent. Importing /wasm explicitly reuses the same fallback class.
module.exports = tranfi.hasNativePreparedTransforms()
  ? require('./factory.js').createPreprocessAPI(() => tranfi, 'native')
  : require('./wasm.js')
