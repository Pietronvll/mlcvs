import numpy as np
from mlcolvar.interpretability.misc import ExplainerOptions
import mlcolvar.interpretability.classification as ic
import mlcolvar.interpretability.regression as ir
from mlcolvar.interpretability.hp_opt import optimize_explainer_model
def explain(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, categorical_labels: bool = False, options: ExplainerOptions = ExplainerOptions()):
    if categorical_labels:
        return explain_boolean(descriptors, labels, feature_names, options = options)
    else:
        return explain_scalar(descriptors, labels, feature_names, options = options)

def explain_scalar(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, options: ExplainerOptions = ExplainerOptions()):
    model, _ = optimize_explainer_model(descriptors, labels, ir.fit_fn, ir.mape, feature_names, options = options, report_progress = True)

def explain_boolean(descriptors: np.ndarray, labels: np.ndarray, feature_names: np.ndarray, options: ExplainerOptions = ExplainerOptions()):
    model, _ = optimize_explainer_model(descriptors, labels, ic.fit_fn, ic.perc_error, feature_names, options = options, report_progress = True)

def partition_categorical(descriptors: np.ndarray, labels: np.ndarray):
    pass