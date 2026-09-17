import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gaze.mapper import GazeMapper
from src.gaze.calibration import CollectedSample
from src.gaze.validation import compute_validation_metrics

def test_validation_metrics():
    mapper = GazeMapper()
    # Dummy fit
    X = np.zeros((1, 10))
    mapper.fit(X, np.array([100]), np.array([100]))
    
    # Mock predict to always return (100, 100)
    # So if target is (103, 104), error is sqrt(3^2 + 4^2) = 5
    
    train_samples = [
        CollectedSample(0, np.zeros(10), 0, 0)
    ]
    
    val_samples = [
        CollectedSample(0, np.zeros(10), 103, 104), # Error 5
        CollectedSample(0, np.zeros(10), 100, 100)  # Error 0
    ]
    
    metrics = compute_validation_metrics(mapper, val_samples, train_samples, 1920, 1080)
    
    assert metrics.validation_targets_excluded_from_fitting is True
    assert metrics.mean_error == 2.5
    assert metrics.max_error == 5.0
    assert metrics.train_targets_count == 1
    assert metrics.val_targets_count == 2

def test_validation_empty():
    mapper = GazeMapper()
    mapper.fit(np.zeros((1,10)), np.array([0]), np.array([0]))
    
    metrics = compute_validation_metrics(mapper, [], [], 1920, 1080)
    assert metrics.val_samples_count == 0
    assert metrics.mean_error == 0.0
