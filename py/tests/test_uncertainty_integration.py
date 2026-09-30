"""Real native predictors exercise the class/property boundary absent in mocks."""
import numpy as np
import pytest
from wlearn import load
from wlearn.liblinear import LinearModel
pytest.importorskip('wlearn_rf')
pytest.importorskip('wlearn_uncertainty')
pytest.importorskip('liblinear')
from wlearn_rf import RFModel
from wlearn_uncertainty import (
    CalibratedClassifier, ConformalClassifier, CrossVennAbersClassifier,
)

X = np.column_stack([np.sin(np.arange(32)), np.cos(np.arange(32))])
y = np.where(X[:, 0] > 0, 7, -3)
Z = X + [.02, -.01]


@pytest.mark.parametrize('cls,params', [(LinearModel, {}), (RFModel, {'n_estimators': 5})])
@pytest.mark.parametrize('wrapper', [CalibratedClassifier, ConformalClassifier])
def test_real_class_axis_and_roundtrip(cls, params, wrapper):
    estimator = cls.create({**params, 'task': 'classification'})
    model = restored = None
    try:
        model = wrapper(estimator=estimator)
        model.fit(X, y)
        model.calibrate(Z, y)
        expected = model.predict_proba(Z)
        restored = load(model.save())
        np.testing.assert_array_equal(restored.predict_proba(Z), expected)
        assert restored.classes() == [-3, 7]
    finally:
        (model or estimator).dispose()
        if restored is not None:
            restored.dispose()


def test_cross_venn_real_class_axis():
    model = CrossVennAbersClassifier(estimator=('linear', LinearModel, {'task': 'classification'}), cv=3)
    restored = None
    try:
        model.fit(X, y)
        restored = load(model.save())
        np.testing.assert_array_equal(restored.predict_proba(Z), model.predict_proba(Z))
    finally:
        model.dispose()
        if restored is not None:
            restored.dispose()


@pytest.mark.parametrize('module,name,params', [
    ('xgboost', 'XGBModel', {'num_round': 3}),
    ('lightgbm', 'LGBModel', {'num_round': 3, 'verbosity': -1}),
    ('stochtree', 'BARTModel', {'num_trees': 3, 'num_gfr': 1, 'num_burnin': 1, 'num_samples': 2}),
])
def test_objective_dependent_capabilities(module, name, params):
    import importlib
    cls = getattr(importlib.import_module('wlearn.' + module), name)
    model = CalibratedClassifier(estimator=cls.create(dict(params, task='classification')))
    restored = None
    labels = (y == 7).astype(np.int32)
    try:
        model.fit(X, labels)
        model.calibrate(Z, labels)
        restored = load(model.save())
        np.testing.assert_allclose(restored.predict_proba(Z), model.predict_proba(Z), rtol=0, atol=1e-5)
    finally:
        model.dispose()
        if restored is not None:
            restored.dispose()
