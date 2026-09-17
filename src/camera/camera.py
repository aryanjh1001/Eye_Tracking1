import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam with index {self.camera_index}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam.")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
