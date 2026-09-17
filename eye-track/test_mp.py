import cv2
import mediapipe as mp
import time
import numpy as np

def test_synthetic():
    print("Testing synthetic image...")
    model_path = "face_landmarker.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )
    
    try:
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Init error: {e}")
        return

    # Create a synthetic image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Must ensure it's contiguous? Let's make it contiguous just in case
    img = np.ascontiguousarray(img)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    start = time.time()
    try:
        print("Calling landmarker.detect()...")
        result = landmarker.detect(mp_image)
        end = time.time()
        print(f"Synthetic detection returned in {end - start:.3f} seconds.")
    except Exception as e:
        print(f"Detect error: {e}")
        
    landmarker.close()

def test_camera():
    print("Testing camera frame loop...")
    model_path = "face_landmarker.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cap = cv2.VideoCapture(0)
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )
    
    try:
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Init error: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
    else:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = np.ascontiguousarray(rgb_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        start = time.time()
        try:
            print("Calling landmarker.detect() on camera frame...")
            result = landmarker.detect(mp_image)
            end = time.time()
            print(f"Camera detection returned in {end - start:.3f} seconds.")
            cv2.imshow("TEST", frame)
            cv2.waitKey(1000)
        except Exception as e:
            print(f"Detect error: {e}")
            
    cap.release()    
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    test_synthetic()
    test_camera()
