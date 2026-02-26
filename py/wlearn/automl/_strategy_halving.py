"""Successive halving strategy matching JS automl/strategy-halving.js."""

import math

from ._rng import make_lcg
from ._sampler import sample_config
from ._common import make_candidate_id


class HalvingStrategy:
    """Multi-round elimination tournament with subsample budgets."""

    def __init__(self, models, n_iter=20, seed=42, factor=3,
                 n_samples=0, greater_is_better=True, cv=5):
        self._factor = factor
        self._n_samples = n_samples
        self._greater_is_better = greater_is_better

        # Generate all candidate configs
        rng = make_lcg(seed)
        all_candidates = []
        for model in models:
            space = model.get('searchSpace') or {}
            if not space and hasattr(model['cls'], 'default_search_space'):
                space = model['cls'].default_search_space()

            effective_space = dict(space)
            fixed_params = model.get('params') or {}
            for key in fixed_params:
                effective_space.pop(key, None)

            config_rng = make_lcg(int(rng() * 0x7fffffff))
            for _ in range(n_iter):
                config = sample_config(effective_space, config_rng)
                params = {**config, **fixed_params}
                candidate_id = make_candidate_id(model['name'], params)
                all_candidates.append({
                    'candidateId': candidate_id,
                    'cls': model['cls'],
                    'params': params,
                })

        self._candidates = all_candidates
        n_cand = len(all_candidates)
        self._n_rounds = max(1, math.ceil(math.log(n_cand) / math.log(factor))) if n_cand > 0 else 1
        self._min_resources = max(cv * 2, math.floor(n_samples / (factor ** self._n_rounds))) if n_samples > 0 else cv * 2

        self._round_index = 0
        self._round = 0
        self._round_results = []
        self._rounds = []
        self._done = False
        self._final_round = False

    def next(self):
        """Return next candidate with subsample budget, or None."""
        if self._done:
            return None
        if self._round_index >= len(self._candidates):
            return None

        cand = self._candidates[self._round_index]
        self._round_index += 1

        if not self._final_round:
            n_resources = min(
                self._n_samples,
                math.floor(self._min_resources * (self._factor ** self._round))
            )
            fraction = min(1, n_resources / self._n_samples) if self._n_samples > 0 else 1
            if fraction < 1:
                return {**cand, 'budget': {'type': 'subsample', 'value': fraction}}

        return cand

    def report(self, result):
        """Report a candidate result. Performs elimination when round is complete."""
        self._round_results.append(result)

        if len(self._round_results) < len(self._candidates):
            return

        if self._final_round:
            self._done = True
            return

        n_resources = min(
            self._n_samples,
            math.floor(self._min_resources * (self._factor ** self._round))
        )
        fraction = min(1, n_resources / self._n_samples) if self._n_samples > 0 else 1

        # Sort results
        sorted_results = list(self._round_results)
        if self._greater_is_better:
            sorted_results.sort(key=lambda r: -r['meanScore'])
        else:
            sorted_results.sort(key=lambda r: r['meanScore'])

        n_survivors = max(1, math.ceil(len(sorted_results) / self._factor))

        self._rounds.append({
            'round': self._round,
            'nResources': n_resources,
            'fraction': fraction,
            'nCandidates': len(self._round_results),
            'nSurvivors': n_survivors,
        })

        # Build survivor candidate list
        survivor_ids = set(
            r['candidateId'] for r in sorted_results[:n_survivors]
        )
        self._candidates = [
            c for c in self._candidates if c['candidateId'] in survivor_ids
        ]

        self._round += 1
        self._round_index = 0
        self._round_results = []

        if len(self._candidates) <= 1 or self._round >= self._n_rounds:
            self._final_round = True

    def is_done(self):
        return self._done

    @property
    def rounds(self):
        return self._rounds
