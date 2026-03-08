const { MockModel } = require('../../ensemble/test/mock-model.js')

/**
 * Mock model that adds defaultSearchSpace() for automl tests.
 * Wraps MockModel from ensemble tests.
 */
class SearchableMock {
  static defaultSearchSpace() {
    return {
      bias: { type: 'uniform', low: -1, high: 1 },
      task: { type: 'categorical', values: ['classification'] },
    }
  }

  static async create(params = {}) {
    return MockModel.create(params)
  }
}

/**
 * Regression variant.
 */
class SearchableMockReg {
  static defaultSearchSpace() {
    return {
      bias: { type: 'uniform', low: -2, high: 2 },
      task: { type: 'categorical', values: ['regression'] },
    }
  }

  static async create(params = {}) {
    return MockModel.create({ task: 'regression', ...params })
  }
}

module.exports = { SearchableMock, SearchableMockReg, MockModel }
