import cv2
import time
import sys
import traceback
import numpy as np
import mediapipe as mp

def print_step(msg):
    print(f"[{time.time():.3f}] {msg}")

def test_a():
    print("\n--- TEST A: Pure OpenCV (Camera -> imshow) ---")
    print(f"UI Framework: {cv2.currentUIFramework()}")
    
    try:
        print_step("Opening camera...")
        cap = cv2.VideoCapture(0)
        
        print_step("Reading frame...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            return
            
        print_step("Creating namedWindow...")
        cv2.namedWindow("Test A", cv2.WINDOW_AUTOSIZE)
        
        print_step("Calling imshow...")
        cv2.imshow("Test A", frame)
        
        print_step("Calling waitKey(1)...")
        key = cv2.waitKey(1)
        
        print_step(f"Window properties: visible={cv2.getWindowProperty('Test A', cv2.WND_PROP_VISIBLE)}")
        print_step(f"Window image rect: {cv2.getWindowImageRect('Test A')}")
        
        print_step("Looping 10 frames...")
        for i in range(10):
            print_step(f"Iteration A-{i} read frame...")
            ret, frame = cap.read()
            if not ret: break
            
            print_step(f"Iteration A-{i} imshow...")
            cv2.imshow("Test A", frame)
            
            print_step(f"Iteration A-{i} waitKey...")
            key = cv2.waitKey(1)
            if key == ord('q'): break
            
        print_step("Cleaning up Test A...")
        cap.release()
        cv2.destroyAllWindows()
        print_step("Test A complete.")
        
    except Exception as e:
        print(f"Exception in Test A:")
        traceback.print_exc()

def test_b():
    print("\n--- TEST B: OpenCV + MediaPipe (Camera -> MP -> imshow) ---")
    try:
        print_step("Initializing MediaPipe...")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        
        print_step("Opening camera...")
        cap = cv2.VideoCapture(0)
        
        print_step("Creating namedWindow...")
        cv2.namedWindow("Test B", cv2.WINDOW_AUTOSIZE)
        
        print_step("Looping 10 frames...")
        for i in range(10):
            print_step(f"Iteration B-{i} read frame...")
            ret, frame = cap.read()
            if not ret: break
            
            print_step(f"Iteration B-{i} BGR -> RGB...")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            
            print_step(f"Iteration B-{i} mp.Image...")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            print_step(f"Iteration B-{i} landmarker.detect()...")
            t1 = time.time()
            result = landmarker.detect(mp_image)
            t2 = time.time()
            print_step(f"Iteration B-{i} detect finished in {t2-t1:.3f}s. Faces: {len(result.face_landmarks)}")
            
            print_step(f"Iteration B-{i} imshow...")
            cv2.imshow("Test B", frame)
            
            print_step(f"Iteration B-{i} waitKey...")
            key = cv2.waitKey(1)
            
            vis = cv2.getWindowProperty("Test B", cv2.WND_PROP_VISIBLE)
            rect = cv2.getWindowImageRect("Test B")
            print_step(f"Iteration B-{i} Window vis={vis}, rect={rect}")
            
            if key == ord('q') or vis < 1: 
                break
                
        print_step("Cleaning up Test B...")
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print_step("Test B complete.")
        
    except Exception as e:
        print(f"Exception in Test B:")
        traceback.print_exc()

if __name__ == "__main__":
    test_a()
    test_b()
