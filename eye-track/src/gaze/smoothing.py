class GazeSmoother:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.smoothed_x = None
        self.smoothed_y = None
        
    def update(self, valid: bool, raw_x: float, raw_y: float):
        if not valid:
            # Hold last valid position
            if self.smoothed_x is None:
                return -1, -1 # No history
            return int(self.smoothed_x), int(self.smoothed_y)
            
        if self.smoothed_x is None:
            self.smoothed_x = raw_x
            self.smoothed_y = raw_y
        else:
            self.smoothed_x = self.alpha * raw_x + (1.0 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * raw_y + (1.0 - self.alpha) * self.smoothed_y
            
        return int(self.smoothed_x), int(self.smoothed_y)
        
    def reset(self):
        self.smoothed_x = None
        self.smoothed_y = None
