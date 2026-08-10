import {
  PLAN_MEDIA_TYPE,
  MinMaxScaler,
  Preprocessor,
  StandardScaler,
  TYPE_ID,
  registerPreprocess
} from '@wlearn/preprocess'

async function compileContract(): Promise<void> {
  const standard = new StandardScaler()
  const minmax = new MinMaxScaler()
  standard.fit([[1], [2]]).transform([[3]])
  minmax.fitTransform([[1], [2]])
  standard.dispose()
  minmax.dispose()
  await registerPreprocess()
  const preprocessor = await Preprocessor.create({
    impute: 'auto',
    encode: 'onehot',
    scale: 'standard',
    maxCategories: 20
  })
  const output = preprocessor.fitTransform({
    dtype: 'float64',
    rows: 2,
    cols: 2,
    data: new Float64Array([1, 2, 2, 3])
  })
  const bytes: Uint8Array = preprocessor.save()
  const restored: Preprocessor = await Preprocessor.load(bytes, {
    limits: { maxApplyRows: 2 }
  })
  restored.transform(output)
  restored.setParams(restored.getParams())
  restored.dispose()
  preprocessor.dispose()
  const typeId: 'wlearn.preprocess.tabular@1' = TYPE_ID
  const mediaType: 'application/x-tranfi-transform-plan' = PLAN_MEDIA_TYPE
  void typeId
  void mediaType
}

void compileContract
