"""Shared construction of the public estimator capability descriptor."""


def estimator_capabilities(*, classifier, regressor, predict_proba,
                           **extra):
    capabilities = {
        'classifier': bool(classifier),
        'regressor': bool(regressor),
        'predictProba': bool(predict_proba),
        'decisionFunction': False,
        'sampleWeight': False,
        'csr': False,
        'earlyStopping': False,
    }
    capabilities.update({key: bool(value) for key, value in extra.items()})
    return capabilities
