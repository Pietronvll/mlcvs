from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class ExplainerOptions:
    train_frac: float = 0.8
    cv_folds: int = 5
    return_all_trials: bool = False
    score_utility_per_feature: float = 1.0

    rng_seed: int = 0 #Reproducibility
    warm_start: float = True

    #HP optimization & search space
    num_trials: int = 10
    min_reg: float = 1e-5
    max_reg: float = 1e4
    sampler: str = 'grid'
    log_sampling: bool = True
    
    #Fitting
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-5

class ExplainerModel:
    def __init__(self, 
                 weights: np.ndarray, 
                 intercept: float, 
                 reg: float, 
                 predict_fn: Callable, 
                 score_fn: Callable, 
                 feature_mean: Optional[np.ndarray] = None, 
                 feature_std: Optional[np.ndarray] = None,
                 feature_names: Optional[np.ndarray] = None
        ):
        self.weights = weights
        self.intercept = intercept
        self.reg = reg
        self.feature_names = np.array([f'x{i}' for i in range(len(weights))]) if feature_names is None else feature_names
        
        self._predict_fn = predict_fn
        self._score_fn = score_fn
        if feature_mean is not None:
            self.feature_mean = feature_mean
        else:
            self.feature_mean = np.zeros_like(weights)
        if feature_std is not None:
            self.feature_std = feature_std
        else:
            self.feature_std = np.ones_like(weights)
    
    @property
    def num_nonzero(self) -> int:
        return np.count_nonzero(self.weights)
    
    @property
    def feature_importance(self) -> np.ndarray:
        _w = (self.weights**2) / np.sum(self.weights**2)
        return [{'name': self.feature_names[i], 'importance': _w[i]} for i in self.nonzero_idxs]
    
    def plot_feature_importance(self):
        vals = [self.weights[i] for i in self.nonzero_idxs]
        names = [self.feature_names[i] for i in self.nonzero_idxs]
        fig, ax = plt.subplots(figsize=(9, 7))
        coefs = pd.DataFrame(
            vals, columns=["Coefficients"], index=names
        )
        ax = coefs.plot(kind="barh", ax=ax, title="Feature importances", legend=False)
        ax.axvline(x=0, color=".5")
        fig.tight_layout()
        return fig, ax
        
    @property
    def nonzero_idxs(self) -> np.ndarray:
        return np.nonzero(self.weights)[0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.feature_mean) / self.feature_std #Scale data before prediction
        return self._predict_fn(X, self.weights, self.intercept)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return self._score_fn(y, y_pred)

def soft_threshold(w: np.ndarray, u: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - u, np.zeros_like(w))

#Todo: add no-grad option
def L1_fista_solver(X: np.ndarray, y: np.ndarray, reg:float, grad_fn:Callable, lipschitz: float, w_init: Optional[np.ndarray] = None, max_iter: int = 100, tol: float = 1e-4):
    #This implementation is highly indebted to https://github.com/scikit-learn-contrib/skglm/blob/52c83195163ef0558eb1dbb1aa7d69491d629466/skglm/solvers/fista.py#L8
    n_samples, n_features = X.shape
    t_new = 1.
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    z = w_init.copy() if w_init is not None else np.zeros(n_features)

    for n_iter in range(max_iter):
        w_old = w.copy()
        grad = grad_fn(X, y, z)
        
        step = 1. / lipschitz
        z -= step * grad
        w = soft_threshold(z, reg * step)

        t_old = t_new
        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        z = w + (t_old - 1.) / t_new * (w - w_old)
        
        #Optimality criterion:
        zero_coeffs = np.maximum(np.abs(grad) - reg, np.zeros_like(w))
        nonzero_coeffs = np.abs(grad + np.sign(w)*reg)
        opt = np.where(w != 0, nonzero_coeffs, zero_coeffs)
        stop_crit = np.max(opt)
        if stop_crit < tol:
            break
    return w, stop_crit, n_iter + 1