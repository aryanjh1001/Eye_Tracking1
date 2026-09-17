import cv2
import time
import os
import sys
import numpy as np
import mediapipe as mp
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.eye_features import EyeFeaturesExtractor

WINDOW_NAME = "Eye Tracking Phase 1"

def main():
    print("Initializing Phase 1...")

    # 1. Initialize MediaPipe EXACTLY as in debug_minimal.py
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Model asset not found at '{model_path}'. Please download it first.")
        return

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # 2. Initialize EyeFeaturesExtractor (pure logic, no threads/GUI)
    eye_extractor = EyeFeaturesExtractor()

    # 3. Initialize Camera EXACTLY as in debug_minimal.py (bypassing Camera class)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        landmarker.close()
        return

    # 4. Create Window EXACTLY as in debug_minimal.py
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Force Window to Foreground using Windows API
    # Call waitKey to process OpenCV's internal window creation messages
    cv2.waitKey(100)
    
    import ctypes
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    
    hwnd = user32.FindWindowW(None, WINDOW_NAME)
    print(f"HWND: {hwnd}")
    if hwnd:
        print(f"IsWindow: {bool(user32.IsWindow(hwnd))}")
        print(f"IsWindowVisible: {bool(user32.IsWindowVisible(hwnd))}")
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.MoveWindow(hwnd, 100, 100, 900, 700, True)
        user32.SetForegroundWindow(hwnd)

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    frame_index = 0

    print("PHASE 1 WINDOW LOOP STARTED")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting loop.")
                break

            # Process frame EXACTLY as in debug_minimal.py
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            result = landmarker.detect(mp_image)

            # Calculate FPS
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()

            face_detected = False
            eyes_detected = False
            iris_detected = False

            if result.face_landmarks and len(result.face_landmarks) > 0:
                face_detected = True
                face_lms = result.face_landmarks[0]
                h, w, _ = frame.shape

                # Extract features
                features = eye_extractor.extract_features(face_lms)

                if len(features['left_eye']) > 0 and len(features['right_eye']) > 0:
                    eyes_detected = True
                if len(features['left_iris']) > 0 and len(features['right_iris']) > 0:
                    iris_detected = True

                # Draw Face landmarks (light gray, small dots)
                for lm in face_lms:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)

                # Draw Eye landmarks (green)
                for lm in features['left_eye'] + features['right_eye']:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Draw Iris landmarks (red)
                for lm in features['left_iris'] + features['right_iris']:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Draw Overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            face_status = "Y" if face_detected else "N"
            eyes_status = "Y" if eyes_detected else "N"
            iris_status = "Y" if iris_detected else "N"
            cv2.putText(frame, f"Face: {face_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {eyes_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Iris: {iris_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            frame_index += 1
            if elapsed_time > 1.0 and fps > 0 and frame_index % 150 == 0:
                print(f"Running... FPS: {fps:.1f}, Frames: {frame_index}")
            
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("q pressed, exiting.")
                break

    except Exception as e:
        print("EXCEPTION IN MAIN LOOP:")
        traceback.print_exc()

    print("Cleaning up...")
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
