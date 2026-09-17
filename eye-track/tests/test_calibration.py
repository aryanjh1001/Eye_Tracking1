import os
import sys
import json
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gaze.calibration import generate_targets, CollectedSample
from gaze_main import save_dataset
import numpy as np

def test_target_generation():
    w, h = 1920, 1080
    targets = generate_targets(w, h)
    
    assert len(targets) == 13
    
    train_t = [t for t in targets if t.is_training]
    val_t = [t for t in targets if not t.is_training]
    
    assert len(train_t) == 9
    assert len(val_t) == 4
    
    # Check disjoint
    train_coords = set((t.x, t.y) for t in train_t)
    val_coords = set((t.x, t.y) for t in val_t)
    
    assert train_coords.isdisjoint(val_coords)
    
def test_csv_generation():
    train_samples = [
        CollectedSample(1.0, np.zeros(10), 100, 100)
    ]
    val_samples = [
        CollectedSample(2.0, np.zeros(10), 200, 200)
    ]
    meta = {"test": True}
    
    csv_path = save_dataset(train_samples, val_samples, meta)
    assert os.path.exists(csv_path)
    
    json_path = csv_path.replace(".csv", ".json")
    assert os.path.exists(json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        assert data["test"] is True
        
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3 # Header + 1 train + 1 val
        assert "split" in lines[0]
        assert "train" in lines[1]
        assert "validation" in lines[2]
