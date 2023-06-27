from typing import Callable, Optional
from mlcolvar.interpretability.misc import ExplainerOptions
import numpy as np
from scipy.stats.qmc import Sobol

def interpretability_loss(num_nonzero: int, error: float, utility_per_feature: float =1.):
    #An error decrease by `utility_per_feature` reduce the loss equally as removing one feature
    if num_nonzero == 0:
        return np.inf #Discard trivial models
    else:
        return error*utility_per_feature + num_nonzero

def k_fold_iterator(num_samples: int, num_folds: int, train_frac:float, rng_seed: int = 0):
    assert 0. < train_frac < 1.
    num_train = int(train_frac*num_samples)

    rng = np.random.default_rng(rng_seed)
    indices = np.arange(num_samples, dtype=np.int32)
    for _ in range(num_folds):
        rng.shuffle(indices)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        yield train_indices, val_indices

def sample_grid(min_val: float, max_val: float, num_samples: int, log_sampling: bool = False):
    if log_sampling:
        sample = np.logspace(np.log10(min_val), np.log10(max_val), num_samples)
    else:
        sample = np.linspace(min_val, max_val, num_samples)
    return sample

def sample_quasimc(min_val: float, max_val: float, num_samples: int, log_sampling: bool = False, rng_seed: int = 0):
    sample_01 = Sobol(dimensions=1, scramble=True, seed=rng_seed).random(num_samples)
    if log_sampling:
        _s = np.log(min_val) + sample_01*np.log(max_val/min_val)
        sample = np.exp(_s)
    else:
        sample = min_val + sample_01*(max_val - min_val)
    return sample

def optimize_explainer_model(X:np.ndarray, y:np.ndarray, fit_fn: Callable, error_fn: Callable, feature_names: Optional[np.ndarray] = None, options: ExplainerOptions = ExplainerOptions(), report_progress: bool = True):
    num_samples, _ = X.shape
    trials = []
    best_trial_idx = 0
    if options.sampler.lower == 'grid':
        reg_samples = sample_grid(options.min_reg, options.max_reg, options.num_trials, options.log_sampling)
    elif options.sampler.lower() == 'quasimc':
        reg_samples = sample_quasimc(options.min_reg, options.max_reg, options.num_trials, options.log_sampling, options.rng_seed)
    else:
        raise ValueError(f'Unknown sampler {options.sampler}. Valid samplers are "grid" and "quasimc"')
    
    for trial_idx, reg in enumerate(reg_samples):
        val_scores = []
        weights = []
        for train_idxs, val_idxs in k_fold_iterator(num_samples, options.cv_folds, options.train_frac, rng_seed = options.rng_seed):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_val, y_val = X[val_idxs], y[val_idxs]
            
            if options.warm_start and trial_idx > 0:
                model = fit_fn(X_train, y_train, reg, options, w_start = trials[trial_idx - 1]['weights'], feature_names = feature_names) #Warm start with the previous model
            else:
                model = fit_fn(X_train, y_train, reg, options, feature_names = feature_names)

            #Validation
            y_pred = model.predict(X_val)
            error = error_fn(y_val, y_pred)
            val_scores.append(interpretability_loss(model.num_nonzero, error, options.score_utility_per_feature))
            weights.append(model.weights)
        avg_val_score = np.mean(val_scores)
        avg_weights = np.mean(np.asarray(weights), axis=0)
        
        if trial_idx > 0:
            best_val_score = trials[best_trial_idx]['score']
            if avg_val_score > best_val_score:
                best_trial_idx = trial_idx
        trials.append({'score': avg_val_score, 'reg': reg, 'weights': avg_weights})
        if report_progress:
            print(f'Trial {trial_idx+1}/{options.num_trials} - reg: {reg:.3e} - best trial: {best_trial_idx + 1} - best score:{trials[best_trial_idx]["score"]:.3e}')

    #Refit the best model on the entire dataset
    model = fit_fn(X, y, trials[best_trial_idx]['reg'], options, w_start = trials[best_trial_idx]['weights'], feature_names = feature_names)

    #Return the best model
    if options.return_all_trials:
        return model, trials
    else:
        return model, trials[best_trial_idx]