from typing import Optional
import numpy as np
import logging
from scipy.special import expit, log_expit
from mlcolvar.interpretability.misc import ExplainerOptions, ExplainerModel, L1_fista_solver

def fit_fn(X: np.ndarray, y: np.ndarray, reg: float, options:ExplainerOptions, w_start: Optional[np.ndarray] = None) -> ExplainerModel:
    n_samples, n_features = X.shape
    #Compute Lipschitz constant
    lipschitz = np.linalg.norm(X, ord = 2) ** 2 / (n_samples * 4.)
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
                accuracy, 
                feature_mean, 
                feature_std)
    return model

#I assume labels in y are 0 and 1
def _transform_labels(y: np.ndarray) -> np.ndarray:
    #From {0, 1} to {-1, 1}
    return 2 * y - 1

def _revert_labels(y: np.ndarray) -> np.ndarray:
    #From {-1, 1} to {0, 1}
    return (y > 0).astype(int)

def predict_fn(X: np.ndarray, weights: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    pred = X @ weights + intercept
    return _revert_labels(pred)

def loss_fn(X: np.ndarray, y: np.ndarray, weights: np.ndarray, intercept: float = 0.0) -> float:
    y_pred = (X @ weights + intercept)
    y_true = _transform_labels(y)
    loss = -log_expit(y_true * y_pred)
    return loss.mean()

def grad_loss_fn(X_and_ones: np.ndarray, y: np.ndarray, weights_and_intercept: np.ndarray) -> np.ndarray:
    y_pred = X_and_ones @ weights_and_intercept
    y = _transform_labels(y)
    return -X_and_ones.T @ (y * expit(-y * y_pred)) #shape (n_features + 1,), 1 for intercept

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

def perc_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 100*(1 - accuracy(y_true, y_pred))