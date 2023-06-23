## Interpretability roadmap

- [ ] Test the FISTA solver on classical benchmarks
- [ ] Add exceptions about `num_iter` and `tol`, if the optimization did not succeed
- [ ] Add tools to extract the feature indices and the feature importance
- [ ] Add tools to partition the classification problems into binary ones
- [x] Add functions to efficiently (warm start) find the optimization path
- [x] Add model selection and validation functions.
- [ ] Add the following hyperparameter sampling schemes (see `optuna`'s source code):
    - [x] Grid search
    - [x] Quasi-Montecarlo low discrepancy sequences
    - [ ] Tree-structured Parzen Estimator
- [ ] Add printing functions to report on the HP optimization status
- [x] Automatically reject models with 0 features (Set validation loss to `np.inf`).
- [ ] Add fitting intercept

### Attributes/methods of the `model` class I should implement:
- [ ] `int: model.num_nonzero`
- [ ] `callable: model.score(X, y) -> float`
- [ ] `callable: model.predict(X) -> np.ndarray`
- [ ] `??: model.feature_importance` structure needs to be defined.

## Notes:
- I have added an explicit `scipy>= 1.8.0` dependency on `pyproject.toml`. However, it should not be an issue, as `KDEpy` already depends on it.