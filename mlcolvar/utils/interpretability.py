from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class ExplainerOptions:
    train_ratio: float = 0.8
    num_folds: int = 5
    return_all_models: bool = False
    score_utility_per_feature: float = 1.0

def explain(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, categorical_labels: bool = False, options: ExplainerOptions = ExplainerOptions()):
    if categorical_labels:
        return explain_categorical(descriptors, labels, feature_names)
    else:
        return explain_scalar(descriptors, labels, feature_names)

def explain_scalar(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, options: ExplainerOptions = ExplainerOptions()):
    _ = get_explainer_model(descriptors, labels, fit_scalar_lasso, options)

def explain_categorical(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, options: ExplainerOptions = ExplainerOptions()):
    datasets = partition_categorical(descriptors, labels)
    for (X, y) in datasets:
        _ = get_explainer_model(X, y, fit_categorical_lasso, options)

def partition_categorical(descriptors: np.ndarray, labels: np.ndarray):
    pass

def get_explainer_model(X:np.ndarray, y:np.ndarray, fit_fn: Callable, val_fn: Callable, options: ExplainerOptions):
    num_samples, _ = X.shape
    seen_models = []
    curr_idx = 0
    best_model_idx = 0
    while not should_stop(seen_models):
        val_scores = []
        reg = suggest_reg(seen_models)
        for train_idxs, val_idxs in cv_split(num_samples, options.num_folds, options.train_ratio):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_val, y_val = X[val_idxs], y[val_idxs]
            model = fit_fn(X_train, y_train, )
            val_scores.append(val_fn(model.num_nonzero, model.score(X_val, y_val), options.score_utility_per_feature))
        avg_val_score = np.mean(val_scores)
            
        if curr_idx > 0:
            best_val_score = seen_models[best_model_idx]['score']
            if avg_val_score > best_val_score:
                best_model_idx = curr_idx
        seen_models.append({'score': avg_val_score, 'reg': reg})
        curr_idx += 1
    if options.return_all_models:
        return seen_models
    else:
        return seen_models[best_model_idx]

def should_stop(seen_models: list):
    pass

def suggest_reg(seen_models: list):
    pass

def categorical_validation_loss(num_nonzero: int, accuracy: float, acc_utility_per_feature: float =1.):
    #Removing one feature has the same utility of adding acc_utility_per_feature% accuracy
    return 100*(1. - accuracy)*acc_utility_per_feature + num_nonzero

def scalar_validation_loss(num_nonzero: int, mape: float, mape_utility_per_feature: float = 1.):
    #Removing one feature has the same utility of reducing the MAPE by mape_utility_per_feature%
    return 100*mape*mape_utility_per_feature + num_nonzero

def cv_split(num_samples: int, num_folds: int, train_ratio:float, rng_seed: int = 0):
    assert 0. < train_ratio < 1.
    num_train = int(train_ratio*num_samples)

    rng = np.random.default_rng(rng_seed)
    indices = np.arange(num_samples, dtype=np.int32)
    for _ in range(num_folds):
        rng.shuffle(indices)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        yield train_indices, val_indices


def fit_scalar_lasso(X: np.ndarray, y: np.ndarray, reg: float = 1.0):
    pass

def fit_categorical_lasso(X: np.ndarray, y: np.ndarray, reg: float = 1.0):
    pass
    