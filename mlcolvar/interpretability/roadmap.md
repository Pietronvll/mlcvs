## Interpretability roadmap

- [ ] Test the FISTA solver on classical benchmarks
- [x] Add warnings about `num_iter` and `tol`, if the optimization did not succeed
- [x] Add tools to extract the feature indices and the feature importance
- [ ] Add tools to partition the classification problems into binary ones
- [x] Add functions to efficiently (warm start) find the optimization path
- [x] Add model selection and validation functions.
- [x] Add the following hyperparameter sampling schemes (see `optuna`'s source code):
    - [x] Grid search
    - [x] Quasi-Montecarlo low discrepancy sequences
- [x] Add printing functions to report on the HP optimization status
- [x] Automatically reject models with 0 features (Set validation loss to `np.inf`).
- [x] Add fitting intercept
### Tests to implement:

#### FISTA solver
1. Lasso regression
2. Lasso classification
3. Warm start
#### `ExplainerModel`:
1. `num_nonzero`
2. `feature_importance`
3. `plot_feature_importance`
4. `nonzero_idxs`
5. `predict`
6. `score`
7. 
#### HP optimization
1. `optimize_explainer_model` with/without feature names
2. `optimize_explainer_model` with/without report progress

#### Regression/Classification
1. `fit_fn`
2. `grad_fn`

### Attributes/methods of the `model` class I should implement:
- [ ] `int: model.num_nonzero`
- [ ] `callable: model.score(X, y) -> float`
- [ ] `callable: model.predict(X) -> np.ndarray`
- [ ] `??: model.feature_importance` structure needs to be defined.

## Notes:
- I have added an explicit `scipy>= 1.8.0` dependency on `pyproject.toml`. However, it should not be an issue, as `KDEpy` already depends on it.