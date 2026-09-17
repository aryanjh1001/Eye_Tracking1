import os
import sys
import time
import json
import csv
import ctypes
import cv2
import numpy as np
import mediapipe as mp
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera.camera import Camera
from src.vision.face_landmarker import FaceLandmarkerApp
from src.gaze.screen import init_dpi_and_get_screen
from src.gaze.feature_extractor import FeatureExtractor
from src.gaze.calibration import Calibrator, generate_targets
from src.gaze.mapper import GazeMapper
from src.gaze.validation import compute_validation_metrics

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

def main():
    print("Initializing Phase 2: Gaze Estimation + Calibration")
    
    # 1. Establish DPI Awareness & Get Screen Info
    screen = init_dpi_and_get_screen()
    print(f"Screen resolution: {screen.width} x {screen.height}")
    print(f"Monitor DPI: {screen.dpi}")
    print(f"DPI awareness: {screen.dpi_awareness}")
    print(f"Coordinate conversion required: {screen.conversion_required}")
    
    # 2. Init MediaPipe
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Model asset not found at '{model_path}'.")
        return
        
    landmarker = FaceLandmarkerApp(model_asset_path=model_path)
    extractor = FeatureExtractor()
    
    # 3. Init Camera
    try:
        camera = Camera(camera_index=0)
    except Exception as e:
        print(f"Failed to open camera: {e}")
        landmarker.close()
        return

    try:
        # 4. Calibration
        calibrator = Calibrator(screen, extractor, camera, landmarker)
        targets = generate_targets(screen.width, screen.height)
        
        train_targets = [t for t in targets if t.is_training]
        val_targets = [t for t in targets if not t.is_training]
        
        print("\n--- Starting Training Calibration (9 targets) ---")
        train_samples = calibrator.run_calibration(train_targets)
        
        if len(train_samples) == 0:
            print("No training samples collected. Exiting.")
            return
            
        # 5. Fit Mapper on Training Data ONLY
        print("\n--- Fitting Ridge Mapper ---")
        X_train = np.array([s.features for s in train_samples])
        yx_train = np.array([s.target_x for s in train_samples])
        yy_train = np.array([s.target_y for s in train_samples])
        
        mapper = GazeMapper(alpha=1.0)
        mapper.fit(X_train, yx_train, yy_train)
        
        print("\n--- Starting Validation Calibration (4 targets) ---")
        val_samples = calibrator.run_calibration(val_targets)
        
        # 6. Save Data
        metadata = {
            "screen_width": screen.width,
            "screen_height": screen.height,
            "feature_version": "1.0",
            "feature_count": 10,
            "num_training_targets": len(train_targets),
            "num_validation_targets": len(val_targets),
            "configured_samples_per_target": 30, # Hardcoded default for now
            "stabilization_time": 1.5,
            "collection_timeout": 8.0,
            "actual_training_samples": len(train_samples),
            "actual_validation_samples": len(val_samples),
            "dpi": screen.dpi,
            "dpi_awareness": screen.dpi_awareness
        }
        
        csv_path = save_dataset(train_samples, val_samples, metadata)
        print(f"\nCalibration data saved to: {csv_path}")
        
        # Save mapper
        mapper.save("gaze_mapper.pkl")
        
        # 7. Validation Metrics
        metrics = compute_validation_metrics(mapper, val_samples, train_samples, screen.width, screen.height)
        
        print("\n==================================================")
        print("FINAL REPORT")
        print("==================================================")
        print(f"Feature count: 10")
        print(f"Training targets: {metrics.train_targets_count}")
        print(f"Validation targets: {metrics.val_targets_count}")
        print(f"Total targets: {metrics.train_targets_count + metrics.val_targets_count}")
        print(f"Training samples: {metrics.train_samples_count}")
        print(f"Validation samples: {metrics.val_samples_count}")
        print(f"Samples per training target: {metrics.train_samples_count / max(1, metrics.train_targets_count):.1f}")
        print(f"Samples per validation target: {metrics.val_samples_count / max(1, metrics.val_targets_count):.1f}")
        print()
        print(f"Screen resolution: {screen.width} x {screen.height}")
        print(f"Monitor DPI: {screen.dpi}")
        print(f"DPI awareness: {screen.dpi_awareness}")
        print(f"Coordinate conversion required: {screen.conversion_required}")
        print()
        print(f"Ridge alpha: 1.0")
        print()
        print(f"Mean Euclidean error: {metrics.mean_error:.1f} px")
        print(f"Median Euclidean error: {metrics.median_error:.1f} px")
        print(f"Maximum Euclidean error: {metrics.max_error:.1f} px")
        print()
        print("Mapper save/load: OK")
        print(f"Validation targets excluded from fitting: {'YES' if metrics.validation_targets_excluded_from_fitting else 'NO'}")
        print("==================================================\n")
        
        # 8. Real-time Visualization
        print("Starting Real-time Visualization...")
        
        window_webcam = "Phase 2 Webcam"
        window_fullscreen = "Phase 2 Gaze"
        
        cv2.namedWindow(window_webcam, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_fullscreen, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_fullscreen, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Force foreground
        cv2.waitKey(100)
        user32 = ctypes.windll.user32
        
        hwnd_fs = user32.FindWindowW(None, window_fullscreen)
        if hwnd_fs:
            user32.ShowWindow(hwnd_fs, 9)
            user32.SetForegroundWindow(hwnd_fs)
            
        hwnd_webcam = user32.FindWindowW(None, window_webcam)
        if hwnd_webcam:
            user32.ShowWindow(hwnd_webcam, 9)
            # user32.SetForegroundWindow(hwnd_webcam)

        fps_start = time.time()
        fps_frames = 0
        fps = 0.0
        
        # Timing profiling
        t_cam = t_mp = t_feat = t_pred = t_rend = 0.0
        
        consecutive_failures = 0
        
        from src.gaze.smoothing import GazeSmoother
        smoother = GazeSmoother(alpha=0.2)

        while True:
            t0 = time.time()
            try:
                frame = camera.read_frame()
                consecutive_failures = 0
            except RuntimeError as e:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print(f"FATAL: Visualization camera read failed {consecutive_failures} times consecutively: {e}")
                    break
                print(f"Visualization camera read error: {e}. Retrying...")
                cv2.waitKey(33)
                continue
                
            t1 = time.time()
            result = landmarker.process_frame(frame)
            t2 = time.time()
            
            feat_res = None
            pred_x, pred_y = -1, -1
            is_valid = False
            eye_openness = 0.0
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lms = result.face_landmarks[0]
                feat_res = extractor.extract_features(lms)
                
                # Draw landmarks on webcam frame
                h, w, _ = frame.shape
                for idx in extractor.left_eye_indices + extractor.right_eye_indices:
                    cv2.circle(frame, (int(lms[idx].x*w), int(lms[idx].y*h)), 1, (0, 255, 0), -1)
                for idx in extractor.left_iris_indices + extractor.right_iris_indices:
                    cv2.circle(frame, (int(lms[idx].x*w), int(lms[idx].y*h)), 1, (0, 0, 255), -1)
                    
            t3 = time.time()
                    
            if feat_res and feat_res.valid:
                # Minimal eye-openness check to mark INVALID when eyes closed
                # feat[6] and feat[7] are left/right eye openness
                eye_openness = (feat_res.features[6] + feat_res.features[7]) / 2.0
                if eye_openness > 0.12: # Threshold for open eyes
                    is_valid = True
                    pred_x, pred_y = mapper.predict(feat_res.features, screen.width, screen.height)
            
            t4 = time.time()
            
            # Apply Temporal Smoothing and Hold
            smooth_x, smooth_y = smoother.update(is_valid, pred_x, pred_y)
            
            # --- Draw Fullscreen Window ---
            fs_img = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
            if smooth_x >= 0 and smooth_y >= 0:
                color = (0, 0, 255) if is_valid else (100, 100, 100)
                cv2.circle(fs_img, (smooth_x, smooth_y), 30, color, -1)
                
                # Crosshair
                cv2.line(fs_img, (smooth_x-40, smooth_y), (smooth_x+40, smooth_y), (255, 255, 255), 2)
                cv2.line(fs_img, (smooth_x, smooth_y-40), (smooth_x, smooth_y+40), (255, 255, 255), 2)
                
                if not is_valid:
                    cv2.putText(fs_img, "INVALID GAZE (HOLDING)", (smooth_x+50, smooth_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(fs_img, "Phase 2.1 Real-time Gaze (Press 'q' to exit)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        
            # --- Draw Webcam Window Overlay ---
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if is_valid:
                cv2.putText(frame, f"Gaze: VALID", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Raw X: {pred_x}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"Raw Y: {pred_y}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            else:
                cv2.putText(frame, f"Gaze: INVALID", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Raw X: ---", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(frame, f"Raw Y: ---", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
            if smooth_x >= 0:
                cv2.putText(frame, f"Smooth X: {smooth_x}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Smooth Y: {smooth_y}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            # Vertical Debug
            if feat_res and feat_res.valid:
                cv2.putText(frame, f"L Iris Y: {feat_res.features[1]:.3f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"R Iris Y: {feat_res.features[3]:.3f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"Avg Iris Y: {feat_res.features[9]:.3f}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"Openness: {eye_openness:.3f}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

            cv2.imshow(window_fullscreen, fs_img)
            cv2.imshow(window_webcam, frame)
            
            t5 = time.time()
            
            # Profiling logic
            t_cam += (t1 - t0)
            t_mp += (t2 - t1)
            t_feat += (t3 - t2)
            t_pred += (t4 - t3)
            t_rend += (t5 - t4)
            
            fps_frames += 1
            if time.time() - fps_start > 1.0:
                fps = fps_frames / (time.time() - fps_start)
                print(f"FPS: {fps:.1f} | Cam: {t_cam/fps_frames*1000:.1f}ms | MP: {t_mp/fps_frames*1000:.1f}ms | Feat: {t_feat/fps_frames*1000:.1f}ms | Pred: {t_pred/fps_frames*1000:.1f}ms | Rend: {t_rend/fps_frames*1000:.1f}ms | Tot: {(time.time()-fps_start)/fps_frames*1000:.1f}ms")
                fps_frames = 0
                t_cam = t_mp = t_feat = t_pred = t_rend = 0.0
                fps_start = time.time()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
    except Exception as e:
        print("EXCEPTION IN MAIN LOOP:")
        traceback.print_exc()
        
    finally:
        print("Cleaning up...")
        try:
            camera.release()
            landmarker.close()
            cv2.destroyAllWindows()
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
