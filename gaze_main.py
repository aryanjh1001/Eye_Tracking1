import sys
import os
import json
import csv
import time
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSlot, QTimer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera.camera import Camera
from src.vision.face_landmarker import FaceLandmarkerApp
from src.gaze.screen import init_dpi_and_get_screen
from src.gaze.feature_extractor import FeatureExtractor
from src.gaze.mapper import GazeMapper
from src.gaze.calibration import generate_targets, CollectedSample
from src.gaze.smoothing import GazeSmoother
from src.gaze.validation import compute_validation_metrics

from src.ui.main_window import MainWindow
from src.ui.overlay import OverlayWidget
from src.ui.camera_thread import CameraThread

def save_dataset(train_samples, val_samples, metadata):
    os.makedirs("data/calibration", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/calibration/calibration_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "split", "target_x", "target_y"] + [f"feat_{i}" for i in range(10)])
        
        for s in train_samples:
            writer.writerow([s.timestamp, "train", s.target_x, s.target_y] + s.features.tolist())
            
        for s in val_samples:
            writer.writerow([s.timestamp, "validation", s.target_x, s.target_y] + s.features.tolist())
            
    json_path = f"data/calibration/calibration_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return csv_path


class AppController:
    def __init__(self):
        # Setup Qt App
        self.app = QApplication(sys.argv)
        
        # Screen info
        self.screen_info = init_dpi_and_get_screen()
        
        # Core components
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        if not os.path.exists(model_path):
            print(f"Error: Model asset not found at '{model_path}'.")
            sys.exit(1)
            
        self.landmarker = FaceLandmarkerApp(model_asset_path=model_path)
        self.extractor = FeatureExtractor()
        
        try:
            self.camera = Camera(camera_index=0)
        except Exception as e:
            print(f"Failed to open camera: {e}")
            sys.exit(1)
            
        self.mapper = GazeMapper(alpha=1.0)
        self.smoother = GazeSmoother(alpha=0.2)
        
        # Calibration state
        self.train_targets = []
        self.val_targets = []
        self.train_samples = []
        self.val_samples = []
        self.is_validating = False
        
        # UI
        self.main_window = MainWindow(self)
        self.overlay = OverlayWidget()
        self.overlay.setGeometry(0, 0, self.screen_info.width, self.screen_info.height)
        self.overlay.calibration_finished.connect(self.on_calibration_finished)
        
        # Background Thread
        self.thread = CameraThread(self.camera, self.landmarker, self.extractor, self.mapper, self.smoother)
        self.thread.set_screen_info(self.screen_info.width, self.screen_info.height)
        
        self.thread.frame_ready.connect(self.main_window.update_frame)
        self.thread.features_extracted.connect(self.on_features)
        self.thread.gaze_predicted.connect(self.overlay.update_gaze)
        
        # Timer to poll calibration buffer
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self.poll_calibration)
        
    def start(self):
        self.main_window.show()
        self.thread.start()
        sys.exit(self.app.exec())
        
    def quit(self):
        self.thread.stop()
        self.landmarker.close()
        self.camera.release()
        self.app.quit()
        
    def start_calibration(self):
        self.main_window.update_status("Calibrating (Training)...")
        targets = generate_targets(self.screen_info.width, self.screen_info.height)
        self.train_targets = [t for t in targets if t.is_training]
        self.val_targets = [t for t in targets if not t.is_training]
        self.train_samples = []
        self.val_samples = []
        
        self.is_validating = False
        
        self.thread.set_calibrating(True)
        self.overlay.start_calibration(self.train_targets)
        self.overlay.showFullScreen()
        self.calib_timer.start(50)
        
    def start_tracking(self):
        self.main_window.update_status("Tracking")
        self.thread.set_calibrating(False)
        self.overlay.mode = "tracking"
        self.overlay.showFullScreen()

    @pyqtSlot(object)
    def on_features(self, feat_res):
        pass # Handling inside poll_calibration for better synchronization
        
    @pyqtSlot()
    def poll_calibration(self):
        if not self.thread._is_calibrating:
            return
            
        if self.overlay.mode != "calibration":
            return
            
        if self.overlay.calibration_state != "collecting":
            self.thread.get_calibration_buffer() # flush buffer while stabilizing
            return
            
        buffer = self.thread.get_calibration_buffer()
        if len(buffer) == 0:
            return
            
        # Add to local lists
        target = self.overlay.targets[self.overlay.current_target_idx]
        for (ts, feats) in buffer:
            sample = CollectedSample(timestamp=ts, features=feats, target_x=target.x, target_y=target.y)
            if self.is_validating:
                self.val_samples.append(sample)
            else:
                self.train_samples.append(sample)
                
        self.overlay.update_calibration_progress(len(buffer))
        
    @pyqtSlot()
    def on_calibration_finished(self):
        self.calib_timer.stop()
        self.thread.set_calibrating(False)
        
        if not self.is_validating:
            # Finished Training, let's fit the model
            if len(self.train_samples) == 0:
                print("No training samples collected.")
                self.overlay.hide()
                self.main_window.update_status("Calibration Failed")
                return
                
            X_train = np.array([s.features for s in self.train_samples])
            yx_train = np.array([s.target_x for s in self.train_samples])
            yy_train = np.array([s.target_y for s in self.train_samples])
            
            self.mapper.fit(X_train, yx_train, yy_train)
            
            # Start validation phase
            self.is_validating = True
            self.main_window.update_status("Calibrating (Validation)...")
            self.thread.set_calibrating(True)
            self.overlay.start_calibration(self.val_targets)
            self.calib_timer.start(50)
            
        else:
            # Finished Validation, Save and show tracking
            self.main_window.update_status("Calibration Complete!")
            
            metadata = {
                "screen_width": self.screen_info.width,
                "screen_height": self.screen_info.height,
                "dpi": self.screen_info.dpi
            }
            save_dataset(self.train_samples, self.val_samples, metadata)
            self.mapper.save("gaze_mapper.pkl")
            
            # Print metrics
            metrics = compute_validation_metrics(self.mapper, self.val_samples, self.train_samples, 
                                                 self.screen_info.width, self.screen_info.height)
            print(f"Calibration Complete! Mean error: {metrics.mean_error:.1f} px")
            
            # Move to tracking mode
            self.start_tracking()

if __name__ == "__main__":
    controller = AppController()
    controller.start()
