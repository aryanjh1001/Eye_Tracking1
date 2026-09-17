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
