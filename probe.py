import cv2
import time
import numpy as np

from src.camera.camera import Camera
from src.vision.face_landmarker import FaceLandmarkerApp
from src.gaze.feature_extractor import FeatureExtractor

def print_feature_formulas():
    print("==================================================")
    print("CURRENT FEATURE FORMULAS")
    print("==================================================")
    print("Stable Corners: Inner=133/362, Outer=33/263")
    print("Eye Width (l_eye_w): max_x - min_x between stable corners")
    print("Inter-eye Dist: distance between left and right eye center points")
    print()
    print("feat_0 [l_iris_x_rel]: (l_iris_x - l_min_x) / l_eye_w")
    print("feat_1 [l_iris_y_rel]: (l_iris_y - l_mid_y) / inter_eye_dist")
    print("feat_2 [r_iris_x_rel]: (r_iris_x - r_min_x) / r_eye_w")
    print("feat_3 [r_iris_y_rel]: (r_iris_y - r_mid_y) / inter_eye_dist")
    print("feat_4 [iris_x_avg]  : (l_iris_x_rel + r_iris_x_rel) / 2.0")
    print("feat_5 [iris_y_avg]  : (l_iris_y_rel + r_iris_y_rel) / 2.0")
    print("feat_6 [l_openness]  : (l_lower.y - l_upper.y) / l_eye_w")
    print("feat_7 [r_openness]  : (r_lower.y - r_upper.y) / r_eye_w")
    print("feat_8 [head_yaw]    : (cheek_left.z - cheek_right.z) / inter_eye_dist")
    print("feat_9 [head_pitch]  : (chin.z - top_head.z) / inter_eye_dist")
    print("==================================================\n")

def run_probe():
    print_feature_formulas()
    
    camera = Camera()
    landmarker = FaceLandmarkerApp()
    extractor = FeatureExtractor()
    
    feature_names = [
        "l_iris_x_rel", "l_iris_y_rel", "r_iris_x_rel", "r_iris_y_rel",
        "iris_x_avg", "iris_y_avg", "l_openness", "r_openness",
        "head_yaw", "head_pitch"
    ]
    num_features = len(feature_names)
    
    # Storage for datasets
    eye_data = {}
    head_data = {}
    baseline_data = None
    
    # State tracking
    current_test_type = "EYE" # "EYE", "BASELINE", "HEAD", "DONE"
    current_label = None
    buffer = []
    target_frames = 0
    
    cv2.namedWindow("LABELED MEASUREMENT PROBE", cv2.WINDOW_AUTOSIZE)
    
    def print_menu():
        print(f"\n--- {current_test_type} TEST MODE ---")
        if current_test_type == "EYE":
            print("Keep HEAD STILL. Move eyes to:")
            print("Press: c=CENTER, l=LEFT, r=RIGHT, u=UP, d=DOWN")
            print("Or press 'b' to advance to BASELINE test once done.")
        elif current_test_type == "BASELINE":
            print("Keep head still, eyes centered.")
            print("Press 'b' to collect 60 frames baseline noise.")
            print("Or press 'h' to advance to HEAD test once done.")
        elif current_test_type == "HEAD":
            print("Keep EYES CENTERED. Move head to:")
            print("Press: c=CENTER, l=LEFT, r=RIGHT, u=UP, d=DOWN")
            print("Or press 'f' to FINISH and print reports.")
    
    print_menu()
    
    try:
        while True:
            try:
                frame = camera.read_frame()
            except RuntimeError as e:
                print(f"Camera error: {e}")
                cv2.waitKey(33)
                continue
                
            result = landmarker.process_frame(frame)
            feat_res = None
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lms = result.face_landmarks[0]
                feat_res = extractor.extract_features(lms)
            
            # HUD overlay
            hud = frame.copy()
            cv2.putText(hud, f"MODE: {current_test_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if feat_res and feat_res.valid:
                cv2.putText(hud, "VALID", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if current_label is not None:
                    buffer.append(feat_res.features)
                    collected = len(buffer)
                    cv2.putText(hud, f"COLLECTING {current_label}: {collected}/{target_frames}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    if collected >= target_frames:
                        # Process collected data
                        data_arr = np.array(buffer)
                        means = np.mean(data_arr, axis=0)
                        medians = np.median(data_arr, axis=0)
                        mins = np.min(data_arr, axis=0)
                        maxs = np.max(data_arr, axis=0)
                        stds = np.std(data_arr, axis=0)
                        
                        print(f"\n==================================================")
                        print(f"{current_test_type} TEST: {current_label}")
                        print(f"Samples: {target_frames}")
                        print(f"==================================================")
                        
                        for i in range(num_features):
                            print(f"\nfeat_{i} {feature_names[i]}")
                            print(f"  mean:   {means[i]:.6f}")
                            print(f"  median: {medians[i]:.6f}")
                            print(f"  min:    {mins[i]:.6f}")
                            print(f"  max:    {maxs[i]:.6f}")
                            print(f"  std:    {stds[i]:.6f}")
                            
                        # Store it
                        stats = {'mean': means, 'median': medians, 'min': mins, 'max': maxs, 'std': stds}
                        if current_test_type == "EYE":
                            eye_data[current_label] = stats
                        elif current_test_type == "BASELINE":
                            baseline_data = stats
                        elif current_test_type == "HEAD":
                            head_data[current_label] = stats
                            
                        current_label = None
                        buffer = []
                        print_menu()
                        
            else:
                cv2.putText(hud, "INVALID", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow("LABELED MEASUREMENT PROBE", hud)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
            if current_label is None: # Only accept input if not collecting
                key_chr = chr(key).lower() if key < 255 else ''
                
                label_map = {'c': 'CENTER', 'l': 'LEFT', 'r': 'RIGHT', 'u': 'UP', 'd': 'DOWN'}
                
                if current_test_type == "EYE":
                    if key_chr in label_map:
                        current_label = label_map[key_chr]
                        buffer = []
                        target_frames = 30
                        print(f"\nStarting {current_test_type} {current_label}...")
                    elif key_chr == 'b':
                        current_test_type = "BASELINE"
                        print_menu()
                elif current_test_type == "BASELINE":
                    if key_chr == 'b':
                        current_label = 'BASELINE'
                        buffer = []
                        target_frames = 60
                        print(f"\nStarting BASELINE noise collection...")
                    elif key_chr == 'h':
                        current_test_type = "HEAD"
                        print_menu()
                elif current_test_type == "HEAD":
                    if key_chr in label_map:
                        current_label = label_map[key_chr]
                        buffer = []
                        target_frames = 30
                        print(f"\nStarting {current_test_type} {current_label}...")
                    elif key_chr == 'f':
                        current_test_type = "DONE"
                        break
                        
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()
        
    if current_test_type == "DONE":
        print_final_reports(feature_names, num_features, eye_data, baseline_data, head_data)

def print_final_reports(names, num, eye_data, baseline_data, head_data):
    if len(eye_data) == 5:
        print("\n==================================================")
        print("EYE-ONLY DELTAS")
        print("==================================================")
        
        c = eye_data['CENTER']['median']
        l = eye_data['LEFT']['median']
        r = eye_data['RIGHT']['median']
        u = eye_data['UP']['median']
        d = eye_data['DOWN']['median']
        
        for i in range(num):
            print(f"\nFeature: feat_{i} [{names[i]}]")
            print(f"  CENTER -> LEFT:  {l[i] - c[i]:.6f}")
            print(f"  CENTER -> RIGHT: {r[i] - c[i]:.6f}")
            print(f"  CENTER -> UP:    {u[i] - c[i]:.6f}")
            print(f"  CENTER -> DOWN:  {d[i] - c[i]:.6f}")
            
        print("\nHORIZONTAL EYE SIGNAL:")
        for i in range(num):
            print(f"  feat_{i}: C->L: {l[i]-c[i]:.6f} | C->R: {r[i]-c[i]:.6f}")
            
        print("\nVERTICAL EYE SIGNAL:")
        for i in range(num):
            print(f"  feat_{i}: C->U: {u[i]-c[i]:.6f} | C->D: {d[i]-c[i]:.6f}")
            
    if baseline_data is not None:
        print("\n==================================================")
        print("CENTER BASELINE NOISE")
        print("==================================================")
        for i in range(num):
            print(f"  feat_{i} std: {baseline_data['std'][i]:.6f}")
            
    if len(head_data) == 5:
        print("\n==================================================")
        print("HEAD-ONLY DELTAS")
        print("==================================================")
        c = head_data['CENTER']['median']
        l = head_data['LEFT']['median']
        r = head_data['RIGHT']['median']
        u = head_data['UP']['median']
        d = head_data['DOWN']['median']
        for i in range(num):
            print(f"\nFeature: feat_{i} [{names[i]}]")
            print(f"  HEAD C -> L: {l[i] - c[i]:.6f}")
            print(f"  HEAD C -> R: {r[i] - c[i]:.6f}")
            print(f"  HEAD C -> U: {u[i] - c[i]:.6f}")
            print(f"  HEAD C -> D: {d[i] - c[i]:.6f}")
            
    if len(eye_data) == 5 and len(head_data) == 5 and baseline_data is not None:
        print("\n==================================================")
        print("EYE VS HEAD COMPARISON (Using Medians)")
        print("==================================================")
        print(f"{'Feature':<15} | {'Eye H-Signal (L-R)':<20} | {'Eye V-Signal (U-D)':<20} | {'Head H-Signal (L-R)':<22} | {'Head V-Signal (U-D)':<22} | {'Center Noise (STD)'}")
        print("-" * 125)
        for i in range(num):
            eye_h = eye_data['LEFT']['median'][i] - eye_data['RIGHT']['median'][i]
            eye_v = eye_data['UP']['median'][i] - eye_data['DOWN']['median'][i]
            head_h = head_data['LEFT']['median'][i] - head_data['RIGHT']['median'][i]
            head_v = head_data['UP']['median'][i] - head_data['DOWN']['median'][i]
            noise = baseline_data['std'][i]
            print(f"feat_{i:<10} | {eye_h:<20.6f} | {eye_v:<20.6f} | {head_h:<22.6f} | {head_v:<22.6f} | {noise:.6f}")

        print("\n==================================================")
        print("SIGNAL-TO-NOISE RATIO (SNR) - ABSOLUTE DELTA / BASELINE STD")
        print("==================================================")
        print(f"{'Feature':<15} | {'Eye H SNR':<12} | {'Eye V SNR':<12} | {'Head H SNR':<12} | {'Head V SNR':<12}")
        print("-" * 75)
        for i in range(num):
            noise = baseline_data['std'][i]
            if noise == 0:
                noise = 1e-6
            eye_h_snr = abs(eye_data['LEFT']['median'][i] - eye_data['RIGHT']['median'][i]) / noise
            eye_v_snr = abs(eye_data['UP']['median'][i] - eye_data['DOWN']['median'][i]) / noise
            head_h_snr = abs(head_data['LEFT']['median'][i] - head_data['RIGHT']['median'][i]) / noise
            head_v_snr = abs(head_data['UP']['median'][i] - head_data['DOWN']['median'][i]) / noise
            print(f"feat_{i:<10} | {eye_h_snr:<12.2f} | {eye_v_snr:<12.2f} | {head_h_snr:<12.2f} | {head_v_snr:<12.2f}")

        print("\n==================================================")
        print("IRIS FEATURES SUMMARY")
        print("==================================================")
        iris_indices = [0, 1, 2, 3, 4, 5]
        for i in iris_indices:
            name = names[i]
            eye_c = eye_data['CENTER']['median'][i]
            eye_l = eye_data['LEFT']['median'][i]
            eye_r = eye_data['RIGHT']['median'][i]
            eye_u = eye_data['UP']['median'][i]
            eye_d = eye_data['DOWN']['median'][i]
            print(f"{name}: Center={eye_c:.4f}, L={eye_l:.4f}, R={eye_r:.4f}, U={eye_u:.4f}, D={eye_d:.4f}")
            
        # Identify strongest signals among iris features
        best_eye_h = -1
        best_eye_h_val = -1
        best_eye_v = -1
        best_eye_v_val = -1
        for i in iris_indices:
            noise = baseline_data['std'][i]
            if noise == 0: noise = 1e-6
            eye_h_snr = abs(eye_data['LEFT']['median'][i] - eye_data['RIGHT']['median'][i]) / noise
            eye_v_snr = abs(eye_data['UP']['median'][i] - eye_data['DOWN']['median'][i]) / noise
            if eye_h_snr > best_eye_h_val:
                best_eye_h_val = eye_h_snr
                best_eye_h = i
            if eye_v_snr > best_eye_v_val:
                best_eye_v_val = eye_v_snr
                best_eye_v = i
                
        print(f"\nStrongest HORIZONTAL eye signal: {names[best_eye_h]} (SNR: {best_eye_h_val:.2f})")
        print(f"Strongest VERTICAL eye signal: {names[best_eye_v]} (SNR: {best_eye_v_val:.2f})")
        print("==================================================\n")

if __name__ == "__main__":
    run_probe()
