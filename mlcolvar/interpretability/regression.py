
from typing import Optional
import logging
import numpy as np
from mlcolvar.interpretability.misc import ExplainerModel, L1_fista_solver, ExplainerOptions

def fit_fn(X: np.ndarray, y: np.ndarray, reg: float, options:ExplainerOptions, w_start: Optional[np.ndarray] = None) -> ExplainerModel:
    n_samples, n_features = X.shape
    #Compute Lipschitz constant
    lipschitz = np.linalg.norm(X, ord = 2) ** 2 / n_samples
    #Scale data
    feature_mean = np.mean(X, axis = 0)
    feature_std = np.std(X, axis = 0)
    X = (X - feature_mean) / feature_std
    #Add intercept if needed
    if options.fit_intercept:
        X = np.concatenate((X, np.ones((n_samples, 1))), axis = 1)
    w, stop_crit, n_iter = L1_fista_solver(X, y, reg, grad_loss_fn, lipschitz, w_init = w_start, max_iter = options.max_iter, tol = options.tol)
    if n_iter == options.max_iter:
        logging.warn(f'FISTA did not converge after {options.max_iter} iterations. Stopped at {stop_crit}.')
    
    if not options.fit_intercept:
        w = np.append(w, 0.0)

    model = ExplainerModel( 
                w[:-1], 
                w[-1], 
                reg, 
                predict_fn, 
                R2, 
                feature_mean, 
                feature_std)
    return model

def predict_fn(X: np.ndarray, weights: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    return X @ weights + intercept

def loss_fn(X: np.ndarray, y: np.ndarray, weights: np.ndarray, intercept: float = 0.0) -> float:
    y_pred = predict_fn(X, weights, intercept)
    return np.mean((y - y_pred)**2)

def grad_loss_fn(X_and_ones: np.ndarray, y: np.ndarray, weights_and_intercept: np.ndarray) -> np.ndarray:
    return -2 * np.mean((y - X_and_ones@weights_and_intercept)[:, np.newaxis] * X_and_ones, axis = 0) #shape (n_features + 1,), 1 for intercept

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return 100*np.mean(np.abs((y_true - y_pred) / (y_true + eps)))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2