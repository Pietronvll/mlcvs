from mlcolvar.interpretability.misc import L1_fista_solver, ExplainerModel
import mlcolvar.interpretability.regression as ir
import mlcolvar.interpretability.classification as ic
import numpy as np
import logging

def make_mock_model(num_features = 50, num_zeros = 20):
    assert num_zeros < num_features
    rng = np.random.default_rng(42)
    w = rng.standard_normal(num_features + 1)
    zero_idxs = np.arange(num_zeros, dtype = np.int64)
    w[zero_idxs] = 0.0
    mock_model = ExplainerModel(
        w[:-1],
        w[-1],
        1.0,
        ir.predict_fn,
        ir.R2
    )
    return mock_model

def test_ExplainerModel_num_nonzero():
    mock_model = make_mock_model(num_zeros = 20)
    assert mock_model.num_nonzero == 20
    
def test_ExplainerModel_nonzero_idxs():
    mock_model = make_mock_model(num_zeros=20)
    assert np.allclose(mock_model.nonzero_idxs, np.arange(20, dtype = np.int64))

def test_ExplainerModel_predict():
    rng = np.random.default_rng(42)
    mock_model = make_mock_model(num_features=50)
    mock_X = rng.standard_normal((100, 50))
    assert np.allclose(mock_model.predict(mock_X), mock_X@(mock_model.weights) + mock_model.intercept)

def test_ExplainerModel_score():
    rng = np.random.default_rng(42)
    mock_model = make_mock_model(num_features=50)
    mock_X = rng.standard_normal((100, 50))
    mock_y = rng.standard_normal(100)
    mock_model.score(mock_X, mock_y)

def test_ExplainerModel_feature_importance():
    mock_model = make_mock_model(num_features = 50, num_zeros = 20)
    feature_importance = mock_model.feature_importance
    assert len(feature_importance) == 30