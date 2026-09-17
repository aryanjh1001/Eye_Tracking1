import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gaze.feature_extractor import FeatureExtractor, FeatureResult
import mediapipe as mp

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def test_feature_extraction_valid():
    extractor = FeatureExtractor()
    
    # Create mock landmarks array of size 478
    lms = [MockLandmark(0.5, 0.5) for _ in range(478)]
    
    # Give specific corners some width/height
    lms[33] = MockLandmark(0.3, 0.3)
    lms[133] = MockLandmark(0.4, 0.3)
    lms[362] = MockLandmark(0.6, 0.3)
    lms[263] = MockLandmark(0.7, 0.3)
    
    # Eyelids for openness
    lms[159] = MockLandmark(0.35, 0.28)
    lms[145] = MockLandmark(0.35, 0.32)
    lms[386] = MockLandmark(0.65, 0.28)
    lms[374] = MockLandmark(0.65, 0.32)
    
    # Irises
    for idx in extractor.left_iris_indices:
        lms[idx] = MockLandmark(0.35, 0.30)
    for idx in extractor.right_iris_indices:
        lms[idx] = MockLandmark(0.65, 0.30)
        
    res = extractor.extract_features(lms)
    assert res.valid is True, f"Failed because: {res.reason}"
    assert res.features is not None
    assert res.features.shape == (10,)
    assert not np.isnan(res.features).any()
    assert not np.isinf(res.features).any()

def test_feature_extraction_missing_face():
    extractor = FeatureExtractor()
    res = extractor.extract_features([])
    assert res.valid is False
    assert "No face" in res.reason

def test_feature_extraction_degenerate_eye():
    extractor = FeatureExtractor()
    lms = [MockLandmark(0.5, 0.5) for _ in range(478)]
    # All points same coordinate = width/height 0
    res = extractor.extract_features(lms)
    assert res.valid is False
    assert "Degenerate" in res.reason or "Near-zero" in res.reason
