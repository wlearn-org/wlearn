"""Random search strategy matching JS automl/strategy-random.js."""

from ._rng import make_lcg
from ._sampler import sample_config
from ._common import make_candidate_id


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

        for model in models:
            space = model.get('searchSpace') or {}
            if not space and hasattr(model['cls'], 'default_search_space'):
                space = model['cls'].default_search_space()

            # Remove fixed params from search space
            effective_space = dict(space)
            fixed_params = model.get('params') or {}
            for key in fixed_params:
                effective_space.pop(key, None)

            config_rng = make_lcg(int(rng() * 0x7fffffff))
            for _ in range(n_iter):
                config = sample_config(effective_space, config_rng)
                params = {**config, **fixed_params}
                candidate_id = make_candidate_id(model['name'], params)
                self._queue.append({
                    'candidateId': candidate_id,
                    'cls': model['cls'],
                    'params': params,
                })

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
