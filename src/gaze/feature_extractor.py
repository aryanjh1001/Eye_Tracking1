import numpy as np
import math
import mediapipe as mp

class FeatureResult:
    def __init__(self, valid: bool, features: np.ndarray = None, reason: str = ""):
        self.valid = valid
        self.features = features
        self.reason = reason

def get_landmark_indices(connections):
    indices = set()
    for connection in connections:
        indices.add(connection.start)
        indices.add(connection.end)
    return list(indices)

class FeatureExtractor:
    def __init__(self):
        connections = mp.tasks.vision.FaceLandmarksConnections
        self.left_eye_indices = get_landmark_indices(connections.FACE_LANDMARKS_LEFT_EYE)
        self.right_eye_indices = get_landmark_indices(connections.FACE_LANDMARKS_RIGHT_EYE)
        self.left_iris_indices = get_landmark_indices(connections.FACE_LANDMARKS_LEFT_IRIS)
        self.right_iris_indices = get_landmark_indices(connections.FACE_LANDMARKS_RIGHT_IRIS)
        
    def _get_centroid(self, face_landmarks, indices):
        if not indices:
            return None
        try:
            x = sum(face_landmarks[i].x for i in indices) / len(indices)
            y = sum(face_landmarks[i].y for i in indices) / len(indices)
            return (x, y)
        except Exception:
            return None

    def _get_bbox(self, face_landmarks, indices):
        if not indices:
            return None
        try:
            xs = [face_landmarks[i].x for i in indices]
            ys = [face_landmarks[i].y for i in indices]
            return min(xs), max(xs), min(ys), max(ys)
        except Exception:
            return None

    def extract_features(self, face_landmarks) -> FeatureResult:
        if not face_landmarks or len(face_landmarks) == 0:
            return FeatureResult(False, None, "No face landmarks")

        try:
            # 1. Iris centroids
            l_iris = self._get_centroid(face_landmarks, self.left_iris_indices)
            r_iris = self._get_centroid(face_landmarks, self.right_iris_indices)

            if not l_iris or not r_iris:
                return FeatureResult(False, None, "Missing iris landmarks")

            l_iris_x, l_iris_y = l_iris
            r_iris_x, r_iris_y = r_iris

            # 2. Stable Eye Corners
            l_inner, l_outer = face_landmarks[362], face_landmarks[263]
            r_inner, r_outer = face_landmarks[133], face_landmarks[33]
            
            l_min_x, l_max_x = min(l_inner.x, l_outer.x), max(l_inner.x, l_outer.x)
            r_min_x, r_max_x = min(r_inner.x, r_outer.x), max(r_inner.x, r_outer.x)
            
            l_mid_y = (l_inner.y + l_outer.y) / 2.0
            r_mid_y = (r_inner.y + r_outer.y) / 2.0
            
            l_eye_w = l_max_x - l_min_x
            r_eye_w = r_max_x - r_min_x
            
            if l_eye_w <= 1e-6 or r_eye_w <= 1e-6:
                return FeatureResult(False, None, "Degenerate eye width")

            # 3. Inter-eye distance using stable corners
            l_center_x, l_center_y = (l_inner.x + l_outer.x) / 2.0, l_mid_y
            r_center_x, r_center_y = (r_inner.x + r_outer.x) / 2.0, r_mid_y
            inter_eye_dist = math.sqrt((l_center_x - r_center_x)**2 + (l_center_y - r_center_y)**2)

            if inter_eye_dist <= 1e-6:
                return FeatureResult(False, None, "Near-zero inter-eye distance")

            # 4. Eye Openness (using mid upper/lower eyelids)
            l_upper, l_lower = face_landmarks[386], face_landmarks[374]
            r_upper, r_lower = face_landmarks[159], face_landmarks[145]
            
            l_openness = (l_lower.y - l_upper.y) / l_eye_w
            r_openness = (r_lower.y - r_upper.y) / r_eye_w

            # 5. Calculate Features
            
            # Iris horizontal positions (normalized between corners)
            l_iris_x_rel = (l_iris_x - l_min_x) / l_eye_w
            r_iris_x_rel = (r_iris_x - r_min_x) / r_eye_w
            
            # Iris vertical positions (distance from corner line, normalized by inter-eye distance)
            l_iris_y_rel = (l_iris_y - l_mid_y) / inter_eye_dist
            r_iris_y_rel = (r_iris_y - r_mid_y) / inter_eye_dist

            # Averaged gaze signals
            iris_x_avg = (l_iris_x_rel + r_iris_x_rel) / 2.0
            iris_y_avg = (l_iris_y_rel + r_iris_y_rel) / 2.0
            
            # Head Pose Proxy (using 3D Z-coordinates)
            # Yaw: difference in Z between left cheek (234) and right cheek (454)
            head_yaw = (face_landmarks[234].z - face_landmarks[454].z) / inter_eye_dist
            # Pitch: difference in Z between chin (152) and top of head (10)
            head_pitch = (face_landmarks[152].z - face_landmarks[10].z) / inter_eye_dist

            features = np.array([
                l_iris_x_rel,
                l_iris_y_rel,
                r_iris_x_rel,
                r_iris_y_rel,
                iris_x_avg,
                iris_y_avg,
                l_openness,
                r_openness,
                head_yaw,
                head_pitch
            ], dtype=np.float32)

            if np.isnan(features).any() or np.isinf(features).any():
                return FeatureResult(False, None, "Features contain NaN or Inf")

            return FeatureResult(True, features, "")
            
        except Exception as e:
            return FeatureResult(False, None, f"Exception during extraction: {e}")
