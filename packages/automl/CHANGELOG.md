# Changelog

## 0.2.0

- Add optional Bayesian search strategy through `@wlearn/bo`
- Depend on the CommonJS `@wlearn/core` and `@wlearn/ensemble` releases
- Add package homepage and GitHub issue metadata

- Add portfolio configs for 8 new model families: rf, mlp, tabm, nam, gam, bart, fm, xlr
- Portfolio now covers 15 model families (up from 7) for both classification and regression

## 0.1.0

- Initial release
- autoFit with RandomSearch, SuccessiveHalvingSearch, PortfolioSearch, ProgressiveSearch
- Caruana ensemble selection, diversity filtering
- Leaderboard with ranking and provenance
- Portfolio configs for 7 model families: xgb, lgb, ebm, linear, svm, knn, tsetlin
- 135 tests
