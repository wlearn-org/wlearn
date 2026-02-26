"""Ensemble methods: voting, stacking, bagging, selection, and OOF predictions."""

from ._voting import VotingEnsemble
from ._stacking import StackingEnsemble
from ._bagging import BaggedEstimator
from ._selection import caruana_select
from ._oof import get_oof_predictions
from ._weights import optimize_weights, project_simplex
