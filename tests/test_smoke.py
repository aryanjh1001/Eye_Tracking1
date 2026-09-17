import os
import sys
import pytest

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera.camera import Camera
from src.vision.face_landmarker import FaceLandmarkerApp

def test_camera_and_processing():
    try:
        # 1. Camera can be opened
        camera = Camera(camera_index=0)
    except Exception as e:
        pytest.skip(f"Camera could not be opened, skipping test. Reason: {e}")

    try:
        # 2. At least one frame can be captured
        frame = camera.read_frame()
        assert frame is not None, "Frame should not be None"
        assert frame.size > 0, "Frame should not be empty"

        # 3. Processing without an exception
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face_landmarker.task")
        
        if not os.path.exists(model_path):
            pytest.skip("Model asset not found, skipping processing test.")

        app = FaceLandmarkerApp(model_asset_path=model_path)
        result = app.process_frame(frame)
        
        # Result should be parsed correctly, even if no face is in view
        assert hasattr(result, 'face_landmarks'), "Result missing face_landmarks attribute"
        
        app.close()
    finally:
        camera.release()
