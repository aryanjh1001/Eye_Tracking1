import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gaze.smoothing import GazeSmoother

def test_smoothing_initial_update():
    smoother = GazeSmoother(alpha=0.2)
    # First valid update should snap directly
    sx, sy = smoother.update(True, 100, 100)
    assert sx == 100
    assert sy == 100
    
def test_smoothing_behavior():
    smoother = GazeSmoother(alpha=0.5)
    smoother.update(True, 100, 100)
    
    # 0.5 * 200 + 0.5 * 100 = 150
    sx, sy = smoother.update(True, 200, 200)
    assert sx == 150
    assert sy == 150
    
    # 0.5 * 300 + 0.5 * 150 = 225
    sx, sy = smoother.update(True, 300, 300)
    assert sx == 225
    assert sy == 225
    
def test_invalid_gaze_holding():
    smoother = GazeSmoother(alpha=0.5)
    smoother.update(True, 100, 100)
    
    # Invalid update should hold previous valid position
    sx, sy = smoother.update(False, 5000, 5000)
    assert sx == 100
    assert sy == 100
    
    # Another invalid update
    sx, sy = smoother.update(False, 0, 0)
    assert sx == 100
    assert sy == 100
    
    # Resume valid
    sx, sy = smoother.update(True, 200, 200)
    # 0.5 * 200 + 0.5 * 100 = 150
    assert sx == 150
    assert sy == 150

def test_initial_invalid_gaze():
    smoother = GazeSmoother(alpha=0.5)
    sx, sy = smoother.update(False, 100, 100)
    assert sx == -1
    assert sy == -1
    
    sx, sy = smoother.update(True, 500, 500)
    assert sx == 500
    assert sy == 500
