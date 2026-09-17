import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle

class GazeMapper:
    def __init__(self, alpha=1.0):
        self.scaler = StandardScaler()
        self.model_x = Ridge(alpha=alpha)
        self.model_y = Ridge(alpha=alpha)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y_x: np.ndarray, y_y: np.ndarray):
        if len(X) == 0:
            raise ValueError("Cannot fit with 0 samples")
        
        # Fit scaler on training data ONLY and transform
        X_scaled = self.scaler.fit_transform(X)
        
        self.model_x.fit(X_scaled, y_x)
        self.model_y.fit(X_scaled, y_y)
        self.is_fitted = True
        
    def predict(self, features: np.ndarray, screen_width: int, screen_height: int) -> tuple[int, int]:
        if not self.is_fitted:
            raise RuntimeError("Mapper is not fitted.")
            
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Transform validation/inference data using the fitted scaler
        features_scaled = self.scaler.transform(features)
            
        pred_x = self.model_x.predict(features_scaled)[0]
        pred_y = self.model_y.predict(features_scaled)[0]
        
        # Clamp to screen bounds
        x = int(np.clip(pred_x, 0, screen_width - 1))
        y = int(np.clip(pred_y, 0, screen_height - 1))
        
        return x, y
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'model_x': self.model_x,
                'model_y': self.model_y,
                'is_fitted': self.is_fitted
            }, f)
            
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        mapper = cls()
        mapper.scaler = data['scaler']
        mapper.model_x = data['model_x']
        mapper.model_y = data['model_y']
        mapper.is_fitted = data['is_fitted']
        return mapper
