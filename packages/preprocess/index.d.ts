import type {
  Labels,
  PreprocessConfig,
  PreprocessResolvedConfig
} from '@wlearn/types'

export interface PreprocessDenseMatrix {
  dtype?: 'float32' | 'float64'
  rows: number
  cols: number
  data: Float32Array | Float64Array
}

export interface PreprocessInputField {
  dtype: 'float64'
  id: string
  name: string
}

export type PreprocessCategoryTag =
  | null
  | { t: 'f32' | 'f64'; v: string }
  | { t: 'other' }

export interface PreprocessOutputField extends PreprocessInputField {
  sourceId: string
  role: 'value' | 'label' | 'onehot'
  category: PreprocessCategoryTag
}

export declare const TYPE_ID: 'wlearn.preprocess.tabular@1'
export declare const PLAN_MEDIA_TYPE: 'application/x-tranfi-transform-plan'

export declare function resolvePreprocessConfig(
  config?: PreprocessConfig | PreprocessResolvedConfig
): Readonly<PreprocessResolvedConfig>

export interface PreprocessRuntimeOptions {
  limits?: Record<string, number>
  cancelFlag?: Int32Array
  cancelToken?: unknown
}

export declare class StandardScaler {
  constructor(params?: Record<string, unknown>)
  fit(X: PreprocessDenseMatrix | number[][]): this
  transform(X: PreprocessDenseMatrix | number[][]): PreprocessDenseMatrix
  fitTransform(X: PreprocessDenseMatrix | number[][]): PreprocessDenseMatrix
  getParams(): Record<string, unknown>
  setParams(params: Record<string, unknown>): this
  save(): Uint8Array
  dispose(): void
  readonly isFitted: boolean
}

export declare class MinMaxScaler extends StandardScaler {}

export declare class Preprocessor {
  private constructor()
  static create(
    config?: PreprocessConfig | PreprocessResolvedConfig,
    runtimeOptions?: PreprocessRuntimeOptions
  ): Promise<Preprocessor>
  static load(
    bytes: Uint8Array | ArrayBuffer,
    runtimeOptions?: PreprocessRuntimeOptions
  ): Promise<Preprocessor>
  fit(X: PreprocessDenseMatrix | number[][], y?: Labels | number[]): this
  transform(X: PreprocessDenseMatrix | number[][]): PreprocessDenseMatrix
  fitTransform(X: PreprocessDenseMatrix | number[][], y?: Labels | number[]): PreprocessDenseMatrix
  getParams(): PreprocessResolvedConfig
  setParams(params: Partial<PreprocessConfig> | PreprocessResolvedConfig): this
  save(): Uint8Array
  dispose(): void
  readonly capabilities: Readonly<{ transformer: true }>
  readonly isFitted: boolean
  readonly backend: 'native' | 'wasm'
  readonly inputSchema: PreprocessInputField[]
  readonly outputSchema: PreprocessOutputField[]
}

export declare function registerPreprocess(options?: {
  backend?: object | Promise<object> | (() => object | Promise<object>)
}): Promise<object>
