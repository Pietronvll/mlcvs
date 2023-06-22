from typing import Optional
import torch

def quad_grad_fn(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Gradient of the function
    .. math:: 1 / (2 xx  n_"samples") ||y - Xw||_2 ^ 2
    '''
    n_samples = y.shape[0]
    return - X.T @ (y - X @ w) / n_samples

def logistic_grad_fn(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Gradient of the function
    .. math:: 1 / n_"samples" \sum_(i=1)^(n_"samples") log(1 + exp(-y_i (Xw)_i))
    '''
    n_samples = y.shape[0]
    return - X.T @ (y * torch.sigmoid(- y * (X @ w))) / n_samples

def soft_thresholding(w: torch.Tensor, u: float) -> torch.Tensor:
    return torch.sign(w) * torch.max(torch.abs(w) - u, torch.zeros_like(w))

#Todo: add no-grad option
def L1_fista_solver(X: torch.Tensor, y: torch.Tensor, reg:float = 1.0, task:str = 'regression',  w_init: Optional[torch.Tensor] = None, max_iter: int = 100, tol: float = 1e-4):
    assert task in ['regression', 'classification'], f"The task {task} is not supported. Please choose between 'regression' and 'classification'"
    if task == 'regression':
        grad_fn = quad_grad_fn
    else:
        grad_fn = logistic_grad_fn

    n_samples, n_features = X.shape
    t_new = 1.
    w = torch.clone(w_init) if w_init is not None else torch.zeros(n_features, device=X.device, dtype = X.dtype)
    z = torch.clone(w_init) if w_init is not None else torch.zeros(n_features, device=X.device, dtype = X.dtype)

    lipschitz = torch.linalg.matrix_norm(X, ord = 2) ** 2 / n_samples
    if task == 'classification':
        lipschitz = lipschitz / 4.

    for n_iter in range(max_iter):
        t_old = t_new
        t_new = (1 + torch.sqrt(1 + 4 * t_old ** 2)) / 2
        w_old = w.clone()
        grad = grad_fn(z, X, y)
        
        step = 1 / lipschitz
        z -= step * grad
        w = soft_thresholding(z, reg * step)
        z = w + (t_old - 1.) / t_new * (w - w_old)
        #Optimality criterion:
        zero_coeffs = torch.maximum(torch.abs(grad) - reg, torch.zeros_like(w))
        nonzero_coeffs = torch.abs(grad + torch.sign(w))*reg
        opt = torch.where(w != 0, nonzero_coeffs, zero_coeffs)
        stop_crit = torch.max(opt)
        if stop_crit < tol:
            break
    return w, stop_crit, n_iter