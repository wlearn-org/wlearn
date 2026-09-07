"""Leaderboard matching JS automl/leaderboard.js."""

from copy import deepcopy
import math

import numpy as np

from ..archive import Archive
from ..errors import ValidationError
from ._candidate import create_candidate, make_candidate_id


def _thaw_domain(value):
    if isinstance(value, dict):
        return {key: _thaw_domain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_domain(item) for item in value]
    return value


class Leaderboard:
    def __init__(self, direction='maximize'):
        if direction not in ('maximize', 'minimize'):
            raise ValidationError('Invalid leaderboard direction')
        self._direction = direction
        self._entries = []
        self._next_id = 0
        self._dirty = True

    def add(self, candidate, scores, fit_time_ms, candidate_id,
            base_seed=42, fold_seeds=None):
        """Add a candidate result.

        Args:
            model_name: str
            params: dict
            scores: np.ndarray of fold scores
            fit_time_ms: float

        Returns:
            entry dict
        """
        normalized_candidate = create_candidate(
            candidate['model'], candidate['model']['params'],
            candidate.get('preprocess'))
        if make_candidate_id(normalized_candidate) != candidate_id:
            raise ValidationError(
                'candidate_id does not match the structured candidate.')

        n = len(scores)
        mean_score = float(np.mean(scores))

        sum_sq = sum((scores[i] - mean_score) ** 2 for i in range(n))
        std_score = math.sqrt(sum_sq / n)

        entry = {
            'id': self._next_id,
            'candidateId': candidate_id,
            'candidate': normalized_candidate,
            'modelName': normalized_candidate['model']['displayName'],
            'params': _thaw_domain(normalized_candidate['model']['params']),
            'scores': np.array(scores, dtype=np.float64),
            'baseSeed': base_seed,
            'foldSeeds': (
                None if fold_seeds is None else
                np.array(fold_seeds, dtype=np.uint32)),
            'meanScore': mean_score,
            'stdScore': std_score,
            'fitTimeMs': fit_time_ms,
            'rank': 0,
            'direction': self._direction,
        }
        self._next_id += 1
        self._entries.append(entry)
        self._dirty = True
        return entry

    def ranked(self):
        """Return all entries sorted by meanScore descending with ranks."""
        if self._dirty:
            self._entries.sort(key=lambda e: e['meanScore'], reverse=self._direction == 'maximize')
            for i, entry in enumerate(self._entries):
                entry['rank'] = i + 1
            self._dirty = False
        return deepcopy(self._entries)

    def best(self):
        """Return the best entry or None."""
        if not self._entries:
            return None
        self.ranked()
        return deepcopy(self._entries[0])

    def top(self, k):
        """Return top k entries."""
        return self.ranked()[:k]

    def to_json(self):
        """Serialize to JSON-friendly list."""
        return [
            {
                'id': e['id'],
                'candidateId': e.get('candidateId'),
                'candidate': create_candidate(
                    e['candidate']['model'],
                    e['candidate']['model']['params'],
                    e['candidate'].get('preprocess')),
                'modelName': e['modelName'],
                'params': deepcopy(e['params']),
                'scores': list(e['scores']),
                'baseSeed': e.get('baseSeed'),
                'foldSeeds': (
                    None if e.get('foldSeeds') is None else
                    [int(value) for value in e['foldSeeds']]),
                'meanScore': e['meanScore'],
                'stdScore': e['stdScore'],
                'fitTimeMs': e['fitTimeMs'],
                'rank': e['rank'],
                'direction': self._direction,
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
                'candidate': deepcopy(e['candidate']),
                'params': deepcopy(e['params']),
                'scores': np.array(e['scores'], dtype=np.float64),
                'foldSeeds': (
                    None if e.get('foldSeeds') is None else
                    np.array(e['foldSeeds'], dtype=np.uint32)),
            })
            if e['id'] >= lb._next_id:
                lb._next_id = e['id'] + 1
        lb._dirty = True
        return lb

    def to_archive(self, metric='score', direction=None, metadata=None):
        """Convert ranked entries to an Archive."""
        archive = Archive(
            id='automl',
            measures=[metric],
            primary_measure=metric,
            direction=direction or self._direction,
            metadata=dict(metadata or {}),
        )
        for entry in self.ranked():
            archive.add({
                'trial_id': f"automl-{entry['id']}",
                'candidate_id': entry['candidateId'],
                'seed': entry.get('baseSeed'),
                'params': deepcopy(entry['candidate']['model']['params']),
                'status': 'ok',
                'scores': {metric: entry['meanScore']},
                'primary_score': entry['meanScore'],
                'timings': {'fitTimeMs': entry['fitTimeMs']},
                'metadata': {
                    'sourceCandidateId': entry['candidateId'],
                    'candidate': deepcopy(entry['candidate']),
                    'leaderboardId': entry['id'],
                    'modelName': entry['modelName'],
                    'foldScores': list(entry['scores']),
                    'foldSeeds': (
                        [] if entry.get('foldSeeds') is None else [
                            {'foldId': fold_id, 'seed': int(value)}
                            for fold_id, value in enumerate(
                                entry['foldSeeds'])
                        ]),
                    'stdScore': entry['stdScore'],
                },
            })
        return archive

    @property
    def length(self):
        return len(self._entries)
