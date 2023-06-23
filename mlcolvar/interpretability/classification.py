import numpy as np
from scipy.special import expit, log_expit

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