"""
Test: namedWindow AFTER first camera read instead of BEFORE.
"""
import cv2
import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera.camera import Camera
from src.vision.face_landmarker import FaceLandmarkerApp

WINDOW_NAME = "Eye Tracking Foundation Phase"

def main():
    print("1. Opening camera...")
    camera = Camera(camera_index=0)
    print("   Camera opened.")

    print("2. Initializing MediaPipe...")
    landmarker = FaceLandmarkerApp(model_asset_path="face_landmarker.task")
    print("   MediaPipe initialized.")

    # DO NOT create namedWindow or call waitKey here.
    # Read the first frame FIRST, then create the window.

    print("3. Reading first frame BEFORE namedWindow...")
    try:
        frame = camera.read_frame()
        print(f"   Frame read OK: shape={frame.shape}")
    except Exception as e:
        print(f"   Frame read FAILED: {e}")
        camera.release()
        landmarker.close()
        return

    print("4. Creating namedWindow AFTER first frame read...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("   namedWindow done.")

    print("5. Processing + displaying...")
    result = landmarker.process_frame(frame)
    print(f"   Faces found: {len(result.face_landmarks) if result.face_landmarks else 0}")
    
    cv2.putText(frame, "Test frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, frame)
    vis = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    print(f"   Visibility after imshow: {vis}")
    cv2.waitKey(1000)

    print("6. Running 20-frame loop...")
    for i in range(20):
        try:
            frame = camera.read_frame()
        except Exception as e:
            print(f"   Frame {i+1} read failed: {e}")
            break
        result = landmarker.process_frame(frame)
        cv2.putText(frame, f"Frame {i+1}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        vis = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if vis < 1:
            print(f"   Window closed at frame {i+1}")
            break
        if (i+1) % 5 == 0:
            print(f"   Frame {i+1}: OK (vis={vis})")

    print("7. Cleanup...")
    camera.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
