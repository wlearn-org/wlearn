"""Progressive evaluation strategy matching JS automl/strategy-progressive.js."""

from ._rng import make_lcg
from ._sampler import sample_config
from ._common import make_candidate_id


class ProgressiveStrategy:
    """Probe all candidates cheaply, then promote top N to full evaluation.

    Phase 1 (probe): yield all candidates with subsample budget
    Phase 2 (promote): yield top N candidates for full evaluation
    """

    def __init__(self, models, n_iter=20, seed=42, promote_count=10,
                 greater_is_better=True, probe_fraction=0.5):
        self._promote_count = promote_count
        self._greater_is_better = greater_is_better
        self._probe_fraction = probe_fraction
        self._phase = 'probe'
        self._probe_index = 0
        self._promote_index = 0
        self._probe_results = []
        self._promoted_candidates = []
        self._done = False

        rng = make_lcg(seed)
        self._all_candidates = []

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
                self._all_candidates.append({
                    'candidateId': candidate_id,
                    'cls': model['cls'],
                    'params': params,
                })

    @property
    def phase(self):
        return self._phase

    def next(self):
        if self._done:
            return None

        if self._phase == 'probe':
            if self._probe_index >= len(self._all_candidates):
                return None
            cand = self._all_candidates[self._probe_index]
            self._probe_index += 1
            if self._probe_fraction < 1:
                return {**cand, 'budget': {'type': 'subsample', 'value': self._probe_fraction}}
            return cand

        # Promote phase
        if self._promote_index >= len(self._promoted_candidates):
            return None
        cand = self._promoted_candidates[self._promote_index]
        self._promote_index += 1
        return cand

    def report(self, result):
        if self._phase == 'probe':
            self._probe_results.append(result)
            if len(self._probe_results) >= len(self._all_candidates):
                self._transition_to_promote()
            return

    def _transition_to_promote(self):
        sorted_results = sorted(
            self._probe_results,
            key=lambda r: r['meanScore'],
            reverse=self._greater_is_better,
        )
        top_n = sorted_results[:max(1, self._promote_count)]
        top_ids = set(r['candidateId'] for r in top_n)
        self._promoted_candidates = [
            c for c in self._all_candidates if c['candidateId'] in top_ids
        ]
        self._phase = 'promote'
        self._promote_index = 0

    def is_done(self):
        if self._done:
            return True
        if (self._phase == 'promote' and
                self._promote_index >= len(self._promoted_candidates) and
                len(self._promoted_candidates) > 0):
            self._done = True
            return True
        return False
