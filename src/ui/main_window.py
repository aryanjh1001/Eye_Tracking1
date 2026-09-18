import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QProgressBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self, app_controller):
        super().__init__()
        self.controller = app_controller
        self.setWindowTitle("Eye Tracking Dashboard")
        self.resize(600, 400)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("Eye Tracking Virtual Keyboard")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Status Layout
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.status_label)
        
        main_layout.addLayout(status_layout)
        
        # Video Feed Preview (Phase 1)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("border: 1px solid #ccc;")
        main_layout.addWidget(self.video_label, stretch=1)
        
        # Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.btn_calibrate = QPushButton("Start Calibration")
        self.btn_calibrate.setMinimumHeight(40)
        self.btn_calibrate.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_calibrate.clicked.connect(self.controller.start_calibration)
        
        self.btn_track = QPushButton("Start Tracking")
        self.btn_track.setMinimumHeight(40)
        self.btn_track.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_track.clicked.connect(self.controller.start_tracking)
        
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setMinimumHeight(40)
        self.btn_quit.clicked.connect(self.controller.quit)
        
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_track)
        btn_layout.addWidget(self.btn_quit)
        
        main_layout.addLayout(btn_layout)
        
        # Apply some styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """)
        
    def update_frame(self, frame):
        # Convert OpenCV BGR frame to PyQt QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # MediaPipe might output RGB or OpenCV reads BGR. Ensure it's RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale pixmap to fit label while keeping aspect ratio
        pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)
        
    def update_status(self, text):
        self.status_label.setText(f"Status: {text}")
        
    def closeEvent(self, event):
        self.controller.quit()
        event.accept()
