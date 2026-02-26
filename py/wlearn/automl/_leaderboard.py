"""Leaderboard matching JS automl/leaderboard.js."""

import math

import numpy as np


class Leaderboard:
    def __init__(self):
        self._entries = []
        self._next_id = 0
        self._dirty = True

    def add(self, model_name, params, scores, fit_time_ms):
        """Add a candidate result.

        Args:
            model_name: str
            params: dict
            scores: np.ndarray of fold scores
            fit_time_ms: float

        Returns:
            entry dict
        """
        n = len(scores)
        mean_score = float(np.mean(scores))

        sum_sq = sum((scores[i] - mean_score) ** 2 for i in range(n))
        std_score = math.sqrt(sum_sq / n)

        entry = {
            'id': self._next_id,
            'modelName': model_name,
            'params': params,
            'scores': np.array(scores, dtype=np.float64),
            'meanScore': mean_score,
            'stdScore': std_score,
            'fitTimeMs': fit_time_ms,
            'rank': 0,
        }
        self._next_id += 1
        self._entries.append(entry)
        self._dirty = True
        return entry

    def ranked(self):
        """Return all entries sorted by meanScore descending with ranks."""
        if self._dirty:
            self._entries.sort(key=lambda e: -e['meanScore'])
            for i, entry in enumerate(self._entries):
                entry['rank'] = i + 1
            self._dirty = False
        return list(self._entries)

    def best(self):
        """Return the best entry or None."""
        if not self._entries:
            return None
        self.ranked()
        return self._entries[0]

    def top(self, k):
        """Return top k entries."""
        return self.ranked()[:k]

    def to_json(self):
        """Serialize to JSON-friendly list."""
        return [
            {
                'id': e['id'],
                'modelName': e['modelName'],
                'params': e['params'],
                'scores': list(e['scores']),
                'meanScore': e['meanScore'],
                'stdScore': e['stdScore'],
                'fitTimeMs': e['fitTimeMs'],
                'rank': e['rank'],
            }
            for e in self.ranked()
        ]

    @classmethod
    def from_json(cls, arr):
        """Deserialize from JSON array."""
        lb = cls()
        for e in arr:
            lb._entries.append({
                **e,
                'scores': np.array(e['scores'], dtype=np.float64),
            })
            if e['id'] >= lb._next_id:
                lb._next_id = e['id'] + 1
        lb._dirty = True
        return lb

    @property
    def length(self):
        return len(self._entries)
