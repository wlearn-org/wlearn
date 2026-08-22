"""Dependency-isolated probes for nanoflann class-label handling."""

import os
import subprocess
import sys
import textwrap


def test_noncontiguous_classifier_labels_use_compact_probability_columns():
    script = textwrap.dedent(
        """
        import sys
        import types

        import numpy as np

        fake = types.ModuleType('pynanoflann')
        fake.KDTree = object
        sys.modules['pynanoflann'] = fake

        from wlearn.nanoflann import KNNModel

        class Tree:
            n_neighbors = 3

            def kneighbors(self, X):
                indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.intp)
                return np.zeros_like(indices, dtype=np.float64), indices

        model = KNNModel(
            np.arange(8, dtype=np.float64).reshape(4, 2),
            np.array([10, 20, 20, 10], dtype=np.int32),
            {'task': 'classification', 'k': 3},
            tree=Tree(),
        )
        query = np.zeros((2, 2), dtype=np.float64)
        predictions = model.predict(query)
        probabilities = model.predict_proba(query).reshape(2, 2)

        assert predictions.dtype == np.int32
        np.testing.assert_array_equal(predictions, [20, 20])
        np.testing.assert_allclose(
            probabilities,
            [[1 / 3, 2 / 3], [1 / 3, 2 / 3]],
        )
        np.testing.assert_array_equal(model.classes, [10, 20])
        """
    )
    environment = dict(os.environ)
    python_path = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        f'py{os.pathsep}{python_path}' if python_path else 'py')
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
