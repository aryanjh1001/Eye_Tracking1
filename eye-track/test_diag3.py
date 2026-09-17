"""
Isolate exactly which initialization step breaks camera reads.
"""
import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_camera_only():
    """Camera opened, read immediately — should work."""
    print("=== TEST 1: Camera only ===")
    cap = cv2.VideoCapture(0)
    print(f"   isOpened: {cap.isOpened()}")
    ret, frame = cap.read()
    print(f"   read: ret={ret}, shape={frame.shape if ret else 'N/A'}")
    cap.release()
    return ret

def test_camera_then_mediapipe():
    """Camera opened, read, THEN init MediaPipe — should work."""
    print("\n=== TEST 2: Camera read THEN MediaPipe init ===")
    cap = cv2.VideoCapture(0)
    print(f"   isOpened: {cap.isOpened()}")
    ret, frame = cap.read()
    print(f"   read before MP: ret={ret}, shape={frame.shape if ret else 'N/A'}")
    
    import mediapipe as mp
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1
        )
    )
    print("   MediaPipe initialized.")

    ret2, frame2 = cap.read()
    print(f"   read after MP: ret={ret2}, shape={frame2.shape if ret2 else 'N/A'}")
    
    landmarker.close()
    cap.release()
    return ret and ret2

def test_mediapipe_then_camera():
    """MediaPipe init, THEN camera open and read."""
    print("\n=== TEST 3: MediaPipe init THEN Camera ===")
    import mediapipe as mp
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1
        )
    )
    print("   MediaPipe initialized.")
    
    cap = cv2.VideoCapture(0)
    print(f"   isOpened: {cap.isOpened()}")
    ret, frame = cap.read()
    print(f"   read: ret={ret}, shape={frame.shape if ret else 'N/A'}")
    
    landmarker.close()
    cap.release()
    return ret

def test_camera_before_mediapipe_read_after():
    """Camera opened BEFORE MediaPipe, read AFTER MediaPipe init."""
    print("\n=== TEST 4: Camera OPEN before MP, Camera READ after MP ===")
    cap = cv2.VideoCapture(0)
    print(f"   isOpened: {cap.isOpened()}")

    import mediapipe as mp
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1
        )
    )
    print("   MediaPipe initialized.")

    ret, frame = cap.read()
    print(f"   read after MP init: ret={ret}, shape={frame.shape if ret else 'N/A'}")
    
    landmarker.close()
    cap.release()
    return ret

if __name__ == "__main__":
    r1 = test_camera_only()
    r2 = test_camera_then_mediapipe()
    r3 = test_mediapipe_then_camera()
    r4 = test_camera_before_mediapipe_read_after()
    
    print("\n=== SUMMARY ===")
    print(f"Test 1 (camera only):              {'PASS' if r1 else 'FAIL'}")
    print(f"Test 2 (camera read then MP):       {'PASS' if r2 else 'FAIL'}")
    print(f"Test 3 (MP then camera):            {'PASS' if r3 else 'FAIL'}")
    print(f"Test 4 (camera open, MP, then read):{'PASS' if r4 else 'FAIL'}")
