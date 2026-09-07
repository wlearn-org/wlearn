"""Internal import aliases; canonical CV and scoring live in wlearn core."""

from ..cv import (
    accuracy, r2_score, neg_mse, neg_mae, neg_logloss, get_scorer,
    k_fold, stratified_k_fold, cross_val_score,
)
