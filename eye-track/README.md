# Eye Tracking Foundation Phase

## Purpose
This is the foundational phase of an eye-tracking virtual keyboard project. In this phase, the application connects to a webcam, captures live video, and uses MediaPipe to detect face, eye, and iris landmarks in real-time. It renders a debug visualization overlay over the webcam feed showing detected regions, status, and FPS.

## Prerequisites
- **Python Version**: 3.14.2 (tested)
- **Dependencies**: Listed in `requirements.txt`. Key libraries include `mediapipe` (Tasks Vision API) and `opencv-python`.

## Setup Instructions

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Asset Setup**:
   This project uses the MediaPipe Tasks Vision Face Landmarker API, which requires the `face_landmarker.task` model asset.
   Download the file directly from Google and place it in the project root:
   ```bash
   python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')"
   ```

## Running the Application

Ensure your webcam is not blocked by another application, then run:
```bash
python main.py
```
Press **`q`** or close the window to exit the application cleanly.

## Smoke Test

To verify the setup without running the full application loop, you can run the included smoke test:
```bash
pytest tests/test_smoke.py
```

## Known Limitations
- The current implementation only tracks and visualizes facial landmarks. Gaze estimation, calibration, and actual OS control are explicitly excluded in this phase.
- Does not contain multiple camera fallback or selection; it defaults to camera index `0`.
- Requires an active webcam and reasonable lighting for the model to correctly identify face and iris landmarks.

## Phase 2: Gaze Estimation & Calibration

Phase 2 builds upon Phase 1 to provide real-time gaze estimation on the screen, without taking control of the operating system cursor.

### Architecture
- **Feature Extraction**: Converts raw MediaPipe landmarks into a compact 10-feature vector containing geometric properties (iris relative position and normalized eye width/height). It uses the center of the eye bounding boxes as a stable inter-eye normalization reference.
- **DPI Awareness**: The application measures DPI and actual screen dimensions (e.g., 1536x960) at runtime to ensure the UI and gaze mapping use correct physical coordinates.
- **13-Target Calibration Design**:
  - **9 Training Targets** placed at 20%, 50%, and 80% screen dimensions.
  - **4 Independent Validation Targets** placed between the training points (35%, 65%).
- **Data Collection**: 
  - Each target uses a **1.5 second stabilization period**.
  - Targets aim to collect approximately **30 valid samples**.
  - An **8 second timeout** prevents indefinite blocking (recording low-confidence targets).
  - Validation samples are strictly additive and are never used to fit the model.
  - Results are saved to a CSV with a JSON metadata sidecar in `data/calibration/`.
- **Ridge Regression**: Fits independent scikit-learn `Ridge(alpha=1.0)` models for X and Y screen coordinates. The mapper supports save/load and clamping.
- **Validation Methodology**: The 4 unseen validation targets are used to compute Mean, Median, and Maximum Euclidean pixel errors.

### Launching Phase 2

To start the Phase 2 calibration and gaze visualization pipeline, run from a standalone PowerShell:

```bash
cd "E:\Eye Tracking Virtual Keyboard\eye-track"
..\venv\Scripts\python.exe gaze_main.py
```

*Note: Phase 2 explicitly does **NOT** control or move the operating-system cursor.*

### Tests
To run all tests for feature extraction, calibration, mapper, validation, and smoke tests:

```bash
..\venv\Scripts\pytest tests\ -v
```
