import time
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

class OverlayWidget(QWidget):
    calibration_finished = pyqtSignal()
    
    def __init__(self, targets=None):
        super().__init__()
        # Make the window frameless and transparent
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # State variables
        self.mode = "tracking" # "tracking" or "calibration"
        self.gaze_x = -1
        self.gaze_y = -1
        self.is_valid = False
        
        # Calibration state
        self.targets = targets or []
        self.current_target_idx = 0
        self.calibration_state = "idle" # "idle", "stabilizing", "collecting"
        self.state_start_time = 0
        self.samples_collected = 0
        self.samples_required = 30
        self.stabilization_time = 1.5
        
        # UI refresh timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16) # ~60fps redraw

    def update_gaze(self, is_valid, x, y):
        self.is_valid = is_valid
        if is_valid:
            self.gaze_x = x
            self.gaze_y = y

    def start_calibration(self, targets):
        self.targets = targets
        self.current_target_idx = 0
        self.mode = "calibration"
        self.advance_target()
        
    def advance_target(self):
        if self.current_target_idx < len(self.targets):
            self.calibration_state = "stabilizing"
            self.state_start_time = time.time()
            self.samples_collected = 0
        else:
            self.mode = "tracking"
            self.calibration_finished.emit()

    def update_calibration_progress(self, new_samples_count):
        self.samples_collected += new_samples_count
        now = time.time()
        
        if self.calibration_state == "stabilizing":
            if now - self.state_start_time >= self.stabilization_time:
                self.calibration_state = "collecting"
                self.state_start_time = now
        elif self.calibration_state == "collecting":
            if self.samples_collected >= self.samples_required or (now - self.state_start_time > 8.0):
                self.current_target_idx += 1
                self.advance_target()
                return True # Target finished
        return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw transparent background (technically handled by WA_TranslucentBackground, 
        # but we can draw a very slight tint if we want, for now completely transparent)
        
        if self.mode == "calibration":
            self._draw_calibration(painter)
        else:
            self._draw_tracking(painter)
            
    def _draw_calibration(self, painter):
        if self.current_target_idx >= len(self.targets): return
        
        target = self.targets[self.current_target_idx]
        
        # Background dimming during calibration
        painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
        
        # Target circle
        color = QColor(0, 255, 0) if self.calibration_state == "collecting" else QColor(255, 255, 0)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(target.x - 20, target.y - 20, 40, 40)
        
        # Draw labels
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
        
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(target.label)
        text_x = max(10, min(target.x - text_width // 2, self.width() - text_width - 10))
        painter.drawText(text_x, target.y - 40, target.label)
        
        if self.calibration_state == "collecting":
            prog_text = f"{self.samples_collected}/{self.samples_required}"
            font.setPointSize(16)
            painter.setFont(font)
            painter.drawText(target.x + 30, target.y + 10, prog_text)
            
    def _draw_tracking(self, painter):
        # Tracking mode is fully transparent background. Only draw the dot.
        if self.gaze_x >= 0 and self.gaze_y >= 0:
            color = QColor(255, 0, 0, 200) if self.is_valid else QColor(100, 100, 100, 200)
            
            # Gaze Dot
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(self.gaze_x - 30), int(self.gaze_y - 30), 60, 60)
            
            # Crosshair
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(int(self.gaze_x - 40), int(self.gaze_y), int(self.gaze_x + 40), int(self.gaze_y))
            painter.drawLine(int(self.gaze_x), int(self.gaze_y - 40), int(self.gaze_x), int(self.gaze_y + 40))
            
            if not self.is_valid:
                painter.setPen(QColor(255, 0, 0))
                painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
                painter.drawText(int(self.gaze_x + 50), int(self.gaze_y), "INVALID GAZE (HOLDING)")
