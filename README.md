<<<<<<< HEAD
# Human Pose Estimation Using Deep Learning

A complete Streamlit application for detecting 17 COCO-format human body keypoints, drawing skeletons, calculating joint angles, classifying simple postures, and estimating repetitions for squats and bicep curls.

> This application is for educational and demonstration purposes. It is not a medical diagnostic or clinical assessment tool.

## 1. Project Introduction

The project combines an Ultralytics pretrained YOLO pose model with OpenCV, NumPy, and Streamlit. It supports image upload, video processing, webcam capture, coordinate tables, angle measurements, posture analysis, and exercise state tracking.

## 2. Problem Statement

Manual body-pose observation is time-consuming and difficult to scale. This application provides an accessible software interface that converts visual frames into keypoint coordinates and interpretable movement measurements.

## 3. Objectives

- Detect people and 17 body keypoints from images, videos, and webcam frames.
- Render a visible skeleton and keypoint overlay.
- Calculate six useful joint angles.
- Provide basic educational posture classification.
- Demonstrate repetition counting using angle state transitions.
- Present results in a modern dashboard.

## 4. Features

- Image upload and original/processed comparison
- Person count and keypoint coordinate table
- Joint-angle analysis and downloadable processed JPG
- Frame-by-frame video processing and MP4 export
- Webcam Start/Stop controls and live skeleton overlay
- Standing, sitting, arms-raised, squatting, and bending heuristics
- Squat and bicep-curl repetition counters
- Low-confidence keypoint handling

## 5. Technologies

- Python 3.11
- Ultralytics YOLO Pose (YOLO26 preferred, YOLO11/YOLOv8 fallback)
- OpenCV
- NumPy
- Streamlit
- Pandas
- Matplotlib
- Pillow
- PyTorch

## 6. System Architecture

```text
app.py
  -> Streamlit sidebar navigation
  -> reusable page modules
  -> pose_model.py
  -> keypoint detection and visualization
  -> angle, posture, and repetition modules
```

The model is cached with Streamlit's `cache_resource` so it is loaded once in each app process. GPU is used automatically when CUDA is available.

## 7. Folder Structure

```text
Human-Pose-Estimation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/pose_model.py
├── modules/
├── pages/
├── utils/
├── assets/style.css
├── dataset/README.md
└── outputs/
```

## 8. Installation on Windows (VS Code)

1. Install Python 3.11 and open VS Code.
2. Open this project folder in VS Code.
3. Open a terminal in the project root.
4. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

5. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver before or after installing requirements.

## 9. Running

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

## 10. How Each Module Works

- `keypoint_detection.py` converts Ultralytics tensors to visible keypoint dictionaries.
- `visualization.py` draws COCO skeleton links and keypoint circles.
- `angle_calculation.py` uses the NumPy vector angle formula.
- `posture_analysis.py` applies conservative geometric thresholds for simple posture categories.
- `exercise_counter.py` uses contraction/extension state transitions to count repetitions.
- `image_processing.py`, `video_processing.py`, and `webcam_processing.py` handle input workflows.

## 11. Expected Outputs

- Annotated RGB image with a skeleton and person boxes
- Original and processed image comparison
- Keypoint coordinate/confidence table
- Joint-angle table in degrees
- Processed MP4 download
- Posture explanation and repetition counts

## 12. Limitations

- Webcam behavior depends on browser permissions and Windows camera privacy settings.
- Pose accuracy changes with lighting, blur, occlusion, clothing, and camera angle.
- Body posture labels are heuristic, not clinically validated.
- Exercise counts may fail with partial joint visibility or unusual camera angles.
- `MOV` decoding may depend on installed OpenCV codecs.

## 13. Future Enhancements

- More pose checkpoint choices
- Multi-person exercise tracking
- Temporal smoothing and Kalman filtering
- Frame-level posture timeline charts
- Saved CSV/JSON reports
- RTMP or IP camera support
- ONNX/TensorRT acceleration
- Persistent multi-person tracking identities
- Multi-camera synchronization

## Advanced Features

- Temporal EMA smoothing for more stable live keypoints
- Session-level confidence, frame, people, posture, and angle analytics
- Joint-angle timeline and posture distribution charts
- JSON session summaries and CSV angle-history exports
- Improved squat/curl repetition state transitions with contraction hold validation
- Exercise form guidance based on measured angles
- Docker deployment configuration

## Docker

```powershell
docker build -t human-pose-estimation .
docker run --rm -p 8501:8501 human-pose-estimation
```

Open `http://localhost:8501`.

## Tests

```powershell
pip install pytest
pytest -q
```
=======
# HUMAN_POSE_ESTIMATION
Human Pose Estimation Project
>>>>>>> a2fd82d61ec84b7a7b07e07f9c151fb771e6049e
