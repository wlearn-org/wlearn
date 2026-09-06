class Step {
  #name
  #estimator

  constructor(name, estimator) {
    this.#name = name
    this.#estimator = estimator
  }

  get name() { return this.#name }
  get estimator() { return this.#estimator }
  get isFitted() { return this.#estimator.isFitted }
  get isTransformer() { return typeof this.#estimator.transform === 'function' }
}

module.exports = { Step }
