# Eye Tracking Virtual Keyboard

A real-time eye-tracking system built with Python, MediaPipe, and OpenCV. Uses webcam-based iris tracking and personal calibration to estimate gaze position on screen.

## Project Structure

```
Eye Tracking Virtual Keyboard/
├── main.py                  # Phase 1: Webcam + MediaPipe landmark visualization
├── gaze_main.py             # Phase 2: Calibration + real-time gaze prediction
├── requirements.txt
├── src/
│   ├── camera/
│   │   └── camera.py        # Webcam capture wrapper
│   ├── vision/
│   │   ├── face_landmarker.py   # MediaPipe FaceLandmarker wrapper
│   │   └── eye_features.py      # Phase 1 eye feature drawing
│   └── gaze/
│       ├── calibration.py       # 13-point calibration (9 train + 4 validation)
│       ├── feature_extractor.py # Stable geometric eye/iris/head-pose features
│       ├── mapper.py            # StandardScaler + Ridge regression gaze mapper
│       ├── screen.py            # DPI-aware screen resolution detection
│       ├── smoothing.py         # Exponential moving average gaze smoother
│       └── validation.py        # Held-out validation metrics
└── tests/
    ├── test_smoke.py
    ├── test_calibration.py
    ├── test_features.py
    ├── test_mapper.py
    ├── test_smoothing.py
    └── test_validation.py
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download `face_landmarker.task` from [MediaPipe Model Cards](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in the project root.

## Running

### Phase 1 — Webcam landmark visualization
```bash
python main.py
```

### Phase 2 — Calibration + real-time gaze
```bash
python gaze_main.py
```
Follow the on-screen calibration targets (9 training + 4 validation). After calibration, the fullscreen gaze window opens showing your predicted gaze point in real time.

### Tests
```bash
pytest tests\ -v
```

## Features

- **MediaPipe FaceMesh** — 478 3D face landmarks at ~30 FPS
- **Stable eye-corner geometry** — iris position normalized between inner/outer corners, decoupled from eyelid movement
- **Head-pose compensation** — yaw and pitch encoded as features so the Ridge model can subtract head movement from gaze signal
- **StandardScaler + Ridge regression** — prevents the center-collapse problem caused by tiny feature variance vs. large screen coordinate scale
- **Exponential Moving Average smoothing** — configurable alpha, smooths the gaze dot without affecting calibration metrics
- **Invalid-gaze holding** — when eyes close or landmarks are unreliable, holds the last valid gaze position
- **PerMonitorV2 DPI awareness** — correct 1920×1200 coordinate mapping on high-DPI displays

## Requirements

- Python 3.10+
- Windows (DPI awareness uses `ctypes.windll`)
- Webcam
