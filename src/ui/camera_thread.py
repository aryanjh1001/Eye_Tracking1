import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    # Signals to communicate with the main thread
    # Emit frame for visualization (e.g. for dashboard)
    frame_ready = pyqtSignal(np.ndarray)
    
    # Emit features when landmarks are found and valid
    features_extracted = pyqtSignal(object) 
    
    # Emit boolean for validity, and predicted (x, y)
    gaze_predicted = pyqtSignal(bool, int, int)
    
    # Emit camera status
    camera_error = pyqtSignal(str)
    
    def __init__(self, camera, landmarker, extractor, mapper=None, smoother=None):
        super().__init__()
        self.camera = camera
        self.landmarker = landmarker
        self.extractor = extractor
        self.mapper = mapper
        self.smoother = smoother
        self.screen_width = 1920 # Defaults, to be updated
        self.screen_height = 1080
        self._is_running = True
        self._is_calibrating = False
        self._current_target = None
        self._calibration_features_buffer = []

    def set_screen_info(self, width, height):
        self.screen_width = width
        self.screen_height = height

    def set_calibrating(self, is_calibrating, current_target=None):
        self._is_calibrating = is_calibrating
        self._current_target = current_target
        if not is_calibrating:
            self._calibration_features_buffer.clear()

    def get_calibration_buffer(self):
        """Consume and return the collected features since last check."""
        buffer = list(self._calibration_features_buffer)
        self._calibration_features_buffer.clear()
        return buffer

    def stop(self):
        self._is_running = False
        self.wait()

    def run(self):
        consecutive_failures = 0
        
        while self._is_running:
            # 1. Read camera
            try:
                frame = self.camera.read_frame()
                consecutive_failures = 0
            except RuntimeError as e:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    self.camera_error.emit(f"Camera read failed consecutively: {e}")
                    break
                time.sleep(0.033)
                continue
                
            # Emit raw frame for dashboard visualization (if needed)
            self.frame_ready.emit(frame)
            
            # 2. Process MediaPipe
            result = self.landmarker.process_frame(frame)
            
            feat_res = None
            pred_x, pred_y = -1, -1
            is_valid = False
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lms = result.face_landmarks[0]
                feat_res = self.extractor.extract_features(lms)
                
            if feat_res and feat_res.valid:
                # Eye openness check
                eye_openness = (feat_res.features[6] + feat_res.features[7]) / 2.0
                if eye_openness > 0.12:
                    is_valid = True
                    self.features_extracted.emit(feat_res)
                    
                    if self._is_calibrating:
                        self._calibration_features_buffer.append((time.time(), feat_res.features))
                    
                    if self.mapper is not None:
                        try:
                            # Map features to gaze
                            pred_x, pred_y = self.mapper.predict(feat_res.features, self.screen_width, self.screen_height)
                        except Exception as e:
                            pass # Mapper might not be fitted yet

            if self.smoother is not None:
                smooth_x, smooth_y = self.smoother.update(is_valid, pred_x, pred_y)
                self.gaze_predicted.emit(is_valid, smooth_x, smooth_y)
            else:
                self.gaze_predicted.emit(is_valid, pred_x, pred_y)

            # Sleep slightly to prevent 100% CPU on loop, usually limited by camera FPS though.
            time.sleep(0.005)
