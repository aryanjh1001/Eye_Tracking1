import mediapipe as mp
import cv2
import numpy as np

class FaceLandmarkerApp:
    def __init__(self, model_asset_path="face_landmarker.task"):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        try:
            self.landmarker = FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Landmarker: {e}")

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Ensure the array is contiguous in memory to prevent C++ binding deadlocks/errors
        rgb_image = np.ascontiguousarray(rgb_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Detect face landmarks
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()
