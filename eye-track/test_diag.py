"""
Diagnostic: reproduce the EXACT initialization and display sequence of main.py
to find why the OpenCV window is not visible.
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
    # Step 1: Camera init (same as main.py)
    print("1. Opening camera...")
    try:
        camera = Camera(camera_index=0)
    except Exception as e:
        print(f"Camera failed: {e}")
        return
    print("   Camera opened.")

    # Step 2: MediaPipe init (same as main.py)
    model_path = "face_landmarker.task"
    print("2. Initializing MediaPipe...")
    landmarker = FaceLandmarkerApp(model_asset_path=model_path)
    print("   MediaPipe initialized.")

    # Step 3: namedWindow (same as main.py)
    print("3. Creating namedWindow with WINDOW_NORMAL...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("   namedWindow done.")

    # Step 4: waitKey pump (same as main.py)
    print("4. cv2.waitKey(1) to pump events...")
    cv2.waitKey(1)
    print("   waitKey done.")

    # Check: is the camera still working?
    print("5. Reading first frame after namedWindow+waitKey...")
    try:
        frame = camera.read_frame()
        print(f"   Frame read OK: shape={frame.shape}, dtype={frame.dtype}")
    except Exception as e:
        print(f"   Frame read FAILED: {e}")
        print("   Attempting direct VideoCapture read...")
        ret, frame = camera.cap.read()
        print(f"   Direct read: ret={ret}")
        if not ret:
            print("   Creating a synthetic frame to test display...")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SYNTHETIC - camera read failed", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Step 6: process with MediaPipe
    print("6. Processing frame with MediaPipe...")
    result = landmarker.process_frame(frame)
    print(f"   Processed. Faces found: {len(result.face_landmarks) if result.face_landmarks else 0}")

    # Step 7: Draw on frame (minimal - just text, no Unicode)
    print("7. Drawing overlay (ASCII only)...")
    cv2.putText(frame, "FPS: 30.0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Face: YES", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Eyes: YES", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Iris: YES", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Step 8: imshow
    print("8. Calling cv2.imshow...")
    vis_pre = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    print(f"   Visibility BEFORE imshow: {vis_pre}")
    
    cv2.imshow(WINDOW_NAME, frame)
    
    vis_post = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    print(f"   Visibility AFTER imshow: {vis_post}")

    # Step 9: waitKey to actually paint the window
    print("9. Calling cv2.waitKey(2000) — window should be visible for 2 seconds...")
    key = cv2.waitKey(2000)
    print(f"   waitKey returned: {key}")

    # Step 10: Now test a tight loop (like main.py does)
    print("10. Running 10-frame tight loop (camera + mediapipe + imshow + waitKey)...")
    for i in range(10):
        try:
            frame = camera.read_frame()
        except Exception:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = landmarker.process_frame(frame)
        cv2.putText(frame, f"Frame {i+1}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print(f"   q pressed at frame {i+1}")
            break
        vis = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if vis < 1:
            print(f"   Window closed at frame {i+1} (vis={vis})")
            break
        print(f"   Frame {i+1}: OK (vis={vis})")

    print("11. Cleanup...")
    camera.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
