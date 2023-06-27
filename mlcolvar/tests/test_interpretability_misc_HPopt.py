from mlcolvar.interpretability.hp_opt import optimize_explainer_model
import mlcolvar.interpretability.regression as ir
import numpy as np
import pytest


optimize_explainer_model(X:np.ndarray, y:np.ndarray, fit_fn: Callable, error_fn: Callable, feature_names: Optional[np.ndarray] = None, options: ExplainerOptions = ExplainerOptions(), report_progress: bool = True):

def mock_regression_data(n_samples = 200, n_features = 100, intercept = 0.0, noise = 0.01):
    assert n_features > 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = ((-1) ** idx) * np.exp(-idx / 10)
    coef[10:] = 0  # sparsify coef
    y = np.dot(X, coef)

    # Add noise
    y += noise * rng.standard_normal(n_samples) + intercept

    # Split data in train set and test set
    X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
    X_test, y_test = X[n_samples // 2 :], y[n_samples // 2 :]
    
    return X_train, y_train, X_test, y_test

def mock_classification_data(n_samples = 200, n_features = 100):
    assert n_features > 10
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))

    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = (-1) ** idx * np.exp(-idx / 10)
    coef[10:] = 0  # sparsify coef
    
    w = np.dot(X, coef)
    prob = 1 / (1 + np.exp(-w))
    y = rng.binomial(1, prob)

    # Split data in train set and test set
    X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
    X_test, y_test = X[n_samples // 2 :], y[n_samples // 2 :]
    
    return X_train, y_train, X_test, y_test

def mock_feature_names(n_features = 100):
    return np.array([f'mock_feature_{i}' for i in range(n_features)])

@pytest.mark.parametrize("log_sampling", [True, False])
@pytest.mark.parametrize("warm_start", [True, False])
@pytest.mark.parametrize("sampler", ["grid", "random"])
@pytest.mark.parametrize("report_progress", [True, False])
@pytest.mark.parametrize("with_feature_names", [True, False])
def test_hpopt_regression(log_sampling, warm_start, sampler, report_progress, with_feature_names):
    options = ExplainerOptions(log_sampling = log_sampling, warm_start = warm_start, sampler = sampler)
    X_train, y_train, X_test, y_test = mock_regression_data()
    if with_feature_names:
        feature_names = mock_feature_names()
    else:
        feature_names = None
    model = optimize_explainer_model(X_train, 
                                     y_train, 
                                     ir.fit_fn, 
                                     ir.mae, 
                                     feature_names = feature_names, 
                                     options = options, 
                                     report_progress = report_progress
    )