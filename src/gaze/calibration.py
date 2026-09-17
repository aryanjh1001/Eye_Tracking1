import cv2
import time
import ctypes
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

SAMPLES_PER_TARGET = 30
STABILIZATION_TIME = 1.5
COLLECTION_TIMEOUT = 8.0

@dataclass
class Target:
    x: int
    y: int
    label: str
    is_training: bool

@dataclass
class CollectedSample:
    timestamp: float
    features: np.ndarray
    target_x: int
    target_y: int

@dataclass
class CalibrationResult:
    training_samples: List[CollectedSample]
    validation_samples: List[CollectedSample]
    metadata: Dict

def generate_targets(screen_width: int, screen_height: int) -> List[Target]:
    targets = []
    
    # 9 training targets (20%, 50%, 80%)
    train_pcts = [0.2, 0.5, 0.8]
    idx = 1
    for py in train_pcts:
        for px in train_pcts:
            targets.append(Target(
                x=int(screen_width * px),
                y=int(screen_height * py),
                label=f"Training Calibration {idx}/9",
                is_training=True
            ))
            idx += 1
            
    # 4 validation targets (35%, 65%)
    val_pcts = [0.35, 0.65]
    idx = 1
    for py in val_pcts:
        for px in val_pcts:
            targets.append(Target(
                x=int(screen_width * px),
                y=int(screen_height * py),
                label=f"Validation {idx}/4",
                is_training=False
            ))
            idx += 1
            
    return targets

class Calibrator:
    def __init__(self, screen_info, feature_extractor, camera, landmarker):
        self.screen = screen_info
        self.extractor = feature_extractor
        self.camera = camera
        self.landmarker = landmarker
        self.window_name = "Phase 2 Calibration"
        
    def _force_foreground(self):
        cv2.waitKey(100)
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, self.window_name)
        if hwnd:
            user32.ShowWindow(hwnd, 9) # SW_RESTORE
            # Center it if not fullscreen, but we are doing fullscreen
            user32.SetForegroundWindow(hwnd)

    def run_calibration(self, targets: List[Target]) -> List[CollectedSample]:
        samples = []
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._force_foreground()
        
        try:
            for target in targets:
                target_samples = []
                state = "stabilizing"
                state_start = time.time()
                
                while True:
                    try:
                        frame = self.camera.read_frame()
                    except RuntimeError:
                        continue
                        
                    # Process frame to get features
                    result = self.landmarker.process_frame(frame)
                    feat_res = None
                    if result.face_landmarks and len(result.face_landmarks) > 0:
                        feat_res = self.extractor.extract_features(result.face_landmarks[0])

                    # UI Drawing
                    display = np.zeros((self.screen.height, self.screen.width, 3), dtype=np.uint8)
                    
                    # Draw target
                    color = (0, 255, 255) # yellow stabilizing
                    if state == "collecting":
                        color = (0, 255, 0) # green collecting
                        
                    cv2.circle(display, (target.x, target.y), 20, color, -1)
                    
                    # Draw label text
                    text_size = cv2.getTextSize(target.label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = target.x - text_size[0] // 2
                    text_y = target.y - 40
                    # Prevent text going off screen
                    text_x = max(10, min(text_x, self.screen.width - text_size[0] - 10))
                    text_y = max(40, text_y)
                    
                    cv2.putText(display, target.label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                    # State machine
                    now = time.time()
                    elapsed = now - state_start
                    
                    if state == "stabilizing":
                        if elapsed >= STABILIZATION_TIME:
                            state = "collecting"
                            state_start = now
                            
                    elif state == "collecting":
                        if feat_res and feat_res.valid:
                            target_samples.append(CollectedSample(
                                timestamp=now,
                                features=feat_res.features,
                                target_x=target.x,
                                target_y=target.y
                            ))
                            
                        # Update UI with collection progress
                        prog_text = f"{len(target_samples)}/{SAMPLES_PER_TARGET}"
                        cv2.putText(display, prog_text, (target.x + 30, target.y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                                    
                        if len(target_samples) >= SAMPLES_PER_TARGET:
                            print(f"Target {target.label} collected {len(target_samples)} samples.")
                            samples.extend(target_samples)
                            break
                            
                        if elapsed >= COLLECTION_TIMEOUT:
                            print(f"WARNING: Target {target.label} timed out after {COLLECTION_TIMEOUT}s. Collected {len(target_samples)}/{SAMPLES_PER_TARGET}.")
                            if len(target_samples) > 0:
                                samples.extend(target_samples)
                            break
                            
                    cv2.imshow(self.window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("Calibration cancelled by user.")
                        cv2.destroyWindow(self.window_name)
                        return samples
                        
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass
                
        return samples
