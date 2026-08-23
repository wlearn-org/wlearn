NPM ?= npm
WLEARN_PYTHON ?= python3
PYTHON ?= $(WLEARN_PYTHON)
Z3_PY ?= $(WLEARN_PYTHON)
export WLEARN_PYTHON

PY_CORE_TESTS = \
	py/tests/test_ecosystem.py \
	py/tests/test_properties.py \
	py/tests/test_automl.py \
	py/tests/test_bundle.py \
	py/tests/test_registry.py \
	py/tests/test_scalers.py \
	py/tests/test_ensemble.py \
	py/tests/test_pipeline_fit.py

.PHONY: test test-js test-browser test-py-core test-py test-z3 test-all

test: test-js test-py-core

test-js:
	$(NPM) test --workspaces --if-present

test-browser:
	$(NPM) run test:browser --workspace @wlearn/ensemble
	$(NPM) run test:browser --workspace @wlearn/automl

test-py-core:
	PYTHONPATH=py $(PYTHON) -m pytest $(PY_CORE_TESTS) -q

test-py:
	PYTHONPATH=py $(PYTHON) -m pytest py/tests -q

test-z3:
	PYTHONPATH=py $(Z3_PY) py/tests/external/resampling_z3.py

test-all: test test-browser test-py
