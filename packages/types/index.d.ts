// wlearn bundle format constants
export declare const BUNDLE_MAGIC: Uint8Array
export declare const BUNDLE_VERSION: 1
export declare const HEADER_SIZE: 16
export declare const DTYPE: {
  readonly FLOAT32: 'float32'
  readonly FLOAT64: 'float64'
  readonly INT32: 'int32'
}

// Data types
export type Dtype = 'float32' | 'float64' | 'int32'

export interface DenseMatrix {
  dtype?: Dtype
  rows: number
  cols: number
  data: Float32Array | Float64Array
}

export interface CSRMatrix {
  dtype?: Dtype
  rows: number
  cols: number
  data: Float64Array
  indices: Int32Array
  indptr: Int32Array
}

export type Matrix = DenseMatrix | CSRMatrix
export type Labels = Int32Array | Float32Array | Float64Array

export interface TensorRef {
  location: 'host' | 'wasm'
  moduleId?: string
  buffer: ArrayBuffer | SharedArrayBuffer
  byteOffset: number
  byteLength: number
  dtype: Dtype
  shape: number[]
  strides?: number[]
}

// Estimator contract
export interface Capabilities {
  classifier: boolean
  regressor: boolean
  predictProba: boolean
  decisionFunction: boolean
  sampleWeight: boolean
  csr: boolean
  earlyStopping: boolean
  [key: string]: boolean
}

export interface Estimator {
  fit(X: Matrix | number[][], y: Labels | number[]): this
  predict(X: Matrix | number[][]): Labels
  score(X: Matrix | number[][], y: Labels | number[]): number
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
}

export interface Classifier extends Estimator {
  predictProba(X: Matrix | number[][]): Float64Array
  readonly classes: Int32Array
}

// Search space IR (for AutoML)
export type SearchParam =
  | { type: 'categorical'; values: unknown[] }
  | { type: 'uniform'; low: number; high: number }
  | { type: 'log_uniform'; low: number; high: number }
  | { type: 'int_uniform'; low: number; high: number }

export type SearchSpace = Record<string, SearchParam & { condition?: Record<string, unknown> }>

// Bundle format v1
export interface BundleManifest {
  typeId: string
  bundleVersion: number
  requires?: string[]
  params?: Record<string, unknown>
  seed?: number
  metadata?: Record<string, unknown>
}

export interface BundleTOCEntry {
  id: string
  offset: number
  length: number
  sha256: string
  mediaType?: string
}

// Pipeline graph (DAG IR for forward compat)
export interface PipelineNode {
  nodeId: string
  typeId: string
  params: Record<string, unknown>
}

export interface PipelineEdge {
  id: string
  from: { nodeId: string; port: string }
  to: { nodeId: string; port: string }
}

export interface PipelineGraph {
  nodes: PipelineNode[]
  edges: PipelineEdge[]
  endpoints: { predict?: string; transform?: string }
}

// Loader
export type LoaderFn = (
  manifest: BundleManifest,
  toc: BundleTOCEntry[],
  blobs: Uint8Array
) => Estimator | Promise<Estimator>
