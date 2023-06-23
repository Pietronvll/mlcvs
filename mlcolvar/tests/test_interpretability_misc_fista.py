from mlcolvar.interpretability.misc import L1_fista_solver
import numpy as np

def mock_regression_data(n_samples = 200, n_features = 100, intercept = 0.0, noise = 0.01):
    assert n_features > 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))

    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = (-1) ** idx * np.exp(-idx / 10)
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