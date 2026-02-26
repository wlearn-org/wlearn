"""AutoML: search space sampling, cross-validation, and hyperparameter search."""

from ._sampler import sample_param, sample_config, random_configs, grid_configs
from ._search import RandomSearch, SuccessiveHalvingSearch
from ._portfolio import PortfolioStrategy, PortfolioSearch, get_portfolio
from ._leaderboard import Leaderboard
from ._auto_fit import auto_fit
from ._executor import Executor
from ._strategy_random import RandomStrategy
from ._strategy_halving import HalvingStrategy
from ._common import detect_task, make_candidate_id, seed_for
from ._cv import (
    k_fold, stratified_k_fold, cross_val_score,
    accuracy, r2_score, neg_mse, neg_mae, get_scorer,
)
from ._rng import make_lcg, shuffle
