from mlcolvar.interpretability.misc import L1_fista_solver
import mlcolvar.interpretability.regression as ir
import mlcolvar.interpretability.classification as ic
import numpy as np
import logging
logging.basicConfig(level = logging.INFO, force=True)

def mock_regression_data(n_samples = 1000, n_features = 100, intercept = 0.0, noise = 0.01):
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

def mock_classification_data(n_samples = 1000, n_features = 100):
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

def test_fista_lasso_regression():
    X_train, y_train, X_test, y_test = mock_regression_data()
    n_samples, n_features = X_train.shape
    w_init = np.zeros_like(X_train[0])
    grad_fn = ir.grad_loss_fn
    lipschitz = (np.linalg.norm(X_train, ord = 2) ** 2) / n_samples
    w, stop_cr, n_iter = L1_fista_solver(X_train, y_train, 0.1, grad_fn, lipschitz, w_init, max_iter = 500, tol = 1e-7)
    assert stop_cr < 1e-7
    assert n_iter < 1000
    assert w.shape == (n_features,)

    idx = np.arange(n_features)
    w_true = ((-1) ** idx) * np.exp(-idx / 10)
    w_true[10:] = 0  # sparsify w_true

    logging.info(f"n_iter = {n_iter}, stop_cr = {stop_cr}, w = {w}, w_true = {w_true}")
    #Testing
    y_pred = ir.predict_fn(X_test, w)
    mse = ir.mse(y_test, y_pred)
    mae = ir.mae(y_test, y_pred)
    mape = ir.mape(y_test, y_pred)
    R2 = ir.R2(y_test, y_pred)
    logging.info(f"MSE: {mse}\nMAE: {mae}MAPE:{mape}\nR2: {R2}")
    logging.info(f"Coefficient error: {np.linalg.norm(w - w_true)}")

def test_fista_lasso_classification():
    X_train, y_train, X_test, y_test = mock_classification_data()
    n_samples, n_features = X_train.shape
    w_init = np.zeros_like(X_train[0])
    grad_fn = ic.grad_loss_fn
    lipschitz = (np.linalg.norm(X_train, ord = 2) ** 2) / (4.*n_samples)
    w, stop_cr, n_iter = L1_fista_solver(X_train, y_train, 0.01, grad_fn, lipschitz, w_init, max_iter = 500)
    assert stop_cr < 1e-4
    assert n_iter < 1000
    assert w.shape == (n_features,)
    idx = np.arange(n_features)
    w_true = ((-1) ** idx) * np.exp(-idx / 10)
    w_true[10:] = 0  # sparsify w_true
    logging.info(f"n_iter = {n_iter}, stop_cr = {stop_cr}, w = {w}, w_true = {w_true}")
    y_pred = ic.predict_fn(X_test, w)
    #Testing
    accuracy = ic.accuracy(y_test, y_pred)
    logging.info(f"Accuracy: {100*accuracy}%")
    idx = np.arange(n_features)
    w_true = ((-1) ** idx) * np.exp(-idx / 10)
    logging.info(f"Coefficient error: {np.linalg.norm(w - w_true)}")