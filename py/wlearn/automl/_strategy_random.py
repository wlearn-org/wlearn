"""Random search strategy matching JS automl/strategy-random.js."""

from ..rng import make_lcg
from ._common import default_search_space
from ._sampler import sample_config
from ._conditions import effective_search_space
from ._candidate import (
    create_candidate_task, normalize_model_specs, preprocess_choices,
)


class RandomStrategy:
    """Generates nIter random configs per model, yields one at a time."""

    def __init__(self, models, n_iter=20, seed=42):
        """
        Args:
            models: list of dicts with 'name', 'cls', optional 'searchSpace', 'params'
            n_iter: candidates per model
            seed: random seed
        """
        self._queue = []
        self._index = 0

        rng = make_lcg(seed)
        seen = {}

        for model in normalize_model_specs(models):
            space = default_search_space(model)

            fixed_params = model.get('params') or {}
            effective_space = effective_search_space(space, fixed_params)

            config_rng = make_lcg(int(rng() * 0x7fffffff))
            for _ in range(n_iter):
                config = sample_config(effective_space, config_rng)
                params = {**config, **fixed_params}
                for preprocess in preprocess_choices(model):
                    self._queue.append(create_candidate_task(
                        model, params, preprocess, seen))

        self._total = len(self._queue)

    def next(self):
        """Return next candidate or None when exhausted."""
        if self._index >= self._total:
            return None
        cand = self._queue[self._index]
        self._index += 1
        return cand

    def report(self, result):
        """No-op for random search."""
        pass

    def is_done(self):
        """True when all candidates have been yielded."""
        return self._index >= self._total
