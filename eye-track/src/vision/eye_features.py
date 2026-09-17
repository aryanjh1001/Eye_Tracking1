import mediapipe as mp

def get_landmark_indices(connections):
    """
    Extracts unique landmark indices from a list of connections.
    Connections are typically tuples of (start_index, end_index).
    """
    indices = set()
    for connection in connections:
        indices.add(connection.start)
        indices.add(connection.end)
    return list(indices)

class EyeFeaturesExtractor:
    def __init__(self):
        # We determine indices dynamically from the API connections
        connections = mp.tasks.vision.FaceLandmarksConnections
        
        self.left_eye_indices = get_landmark_indices(connections.FACE_LANDMARKS_LEFT_EYE)
        self.right_eye_indices = get_landmark_indices(connections.FACE_LANDMARKS_RIGHT_EYE)
        
        self.left_iris_indices = get_landmark_indices(connections.FACE_LANDMARKS_LEFT_IRIS)
        self.right_iris_indices = get_landmark_indices(connections.FACE_LANDMARKS_RIGHT_IRIS)

    def extract_features(self, face_landmarks):
        """
        Extracts specific landmark coordinates given the full face landmarks.
        Returns a dictionary containing left_eye, right_eye, left_iris, right_iris landmarks.
        """
        return {
            'left_eye': [face_landmarks[i] for i in self.left_eye_indices],
            'right_eye': [face_landmarks[i] for i in self.right_eye_indices],
            'left_iris': [face_landmarks[i] for i in self.left_iris_indices],
            'right_iris': [face_landmarks[i] for i in self.right_iris_indices],
        }
