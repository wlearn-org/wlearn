// wlearn bundle format constants
export const BUNDLE_MAGIC = new Uint8Array([0x57, 0x4c, 0x52, 0x4e]) // 'WLRN'
export const BUNDLE_VERSION = 1
export const HEADER_SIZE = 16 // magic(4) + version(4) + manifestLen(4) + tocLen(4)
export const DTYPE = { FLOAT32: 'float32', FLOAT64: 'float64', INT32: 'int32' }
