import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gaze.mapper import GazeMapper

def test_mapper_training_and_prediction():
    mapper = GazeMapper()
    
    # Synthetic data
    X = np.random.rand(10, 10)
    y_x = np.random.randint(0, 1920, size=10)
    y_y = np.random.randint(0, 1080, size=10)
    
    mapper.fit(X, y_x, y_y)
    assert mapper.is_fitted
    
    # Scaler must be fitted
    assert hasattr(mapper.scaler, 'mean_')
    assert mapper.scaler.mean_ is not None
    
    features = np.random.rand(10)
    pred_x, pred_y = mapper.predict(features, 1920, 1080)
    
    assert isinstance(pred_x, int)
    assert isinstance(pred_y, int)
    assert np.isfinite(pred_x) and np.isfinite(pred_y)
    
    # Clamping test
    y_x_huge = np.array([5000]*10)
    y_y_huge = np.array([5000]*10)
    mapper.fit(X, y_x_huge, y_y_huge)
    px, py = mapper.predict(features, 1920, 1080)
    assert px <= 1919
    assert py <= 1079
    
def test_regression_diagnostic_center_collapse():
    # Deterministic test to verify standardization prevents predicting center
    mapper = GazeMapper(alpha=1.0)
    
    # Tiny feature variance like real data (stdev 0.02)
    # Make feature 0 linearly correlated with X, feature 1 with Y
    x_vals = np.linspace(0.4, 0.6, 20)
    y_vals = np.linspace(0.4, 0.6, 20)
    
    X = np.zeros((20, 10))
    X[:, 0] = x_vals
    X[:, 1] = y_vals
    
    # Coordinates spreading out to 1920
    y_x = np.linspace(0, 1920, 20)
    y_y = np.linspace(0, 1080, 20)
    
    mapper.fit(X, y_x, y_y)
    
    # Pick the extreme feature that maps to 1920
    px, py = mapper.predict(X[-1], 1920, 1080)
    
    # If unscaled, it would collapse to mean (960). With scaler, it should be much closer to 1920.
    assert px > 1500, f"Predicted X ({px}) collapsed towards mean (960) instead of expected ~1920"

def test_mapper_unfitted():
    mapper = GazeMapper()
    with pytest.raises(RuntimeError):
        mapper.predict(np.zeros(10), 1920, 1080)

def test_mapper_save_load():
    mapper = GazeMapper()
    X = np.random.rand(10, 10)
    y_x = np.random.randint(0, 1920, size=10)
    y_y = np.random.randint(0, 1080, size=10)
    mapper.fit(X, y_x, y_y)
    
    path = "test_mapper.pkl"
    mapper.save(path)
    
    assert os.path.exists(path)
    
    loaded = GazeMapper.load(path)
    assert loaded.is_fitted
    assert hasattr(loaded.scaler, 'mean_')
    
    feat = np.random.rand(10)
    p1 = mapper.predict(feat, 1920, 1080)
    p2 = loaded.predict(feat, 1920, 1080)
    
    assert p1 == p2
    
    if os.path.exists(path):
        os.remove(path)
