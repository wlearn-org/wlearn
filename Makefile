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
	py/tests/test_composite_load_hardening.py \
	py/tests/test_pipeline_fit.py

.NOTPARALLEL: test-all

.PHONY: test test-js test-browser test-py-core test-py test-z3 test-all

test: test-js test-py-core test-release

test-js:
	$(NPM) run test:js

test-browser:
	$(NPM) run test:browser

test-py-core:
	PYTHONPATH=py $(PYTHON) -m pytest $(PY_CORE_TESTS) -q

test-py:
	PYTHONPATH=py $(PYTHON) -m pytest py/tests -q

test-z3:
	PYTHONPATH=py $(Z3_PY) py/tests/external/resampling_z3.py

test-all: test-release test-js test-types test-browser test-py test-integration test-interop

.PHONY: test-integration
test-integration:
	$(NPM) run test:integration

.PHONY: test-types test-interop
test-types:
	$(NPM) run test:types

test-interop:
	WLEARN_PYTHON=$(PYTHON) $(NPM) run test:interop:full

.PHONY: test-release
test-release:
	$(PYTHON) -m unittest discover -s scripts -p test_release.py -v
