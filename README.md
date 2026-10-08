# Advanced Crowd Detection System

A complete computer-vision project for detecting and tracking people in live video, webcam streams, or saved videos.

## Features

- Person detection using YOLOv8
- Real-time multi-object tracking
- Live crowd count
- Region-of-interest (ROI) filtering
- Crowd density classification
- Overcrowding alerts
- Alert cooldown to avoid repeated triggers
- Snapshot capture during alerts
- CSV event logging
- FPS display
- Unique-person counting
- Entry/exit line crossing counts
- Webcam and video-file support
- Optional annotated video recording
- Configurable confidence, thresholds, ROI, line position, and alert settings
- Modular Python project structure

## Important use note

This project is intended for general crowd-safety, occupancy, event-management, and learning use.
It detects people as anonymous objects and does not perform face recognition or identity matching.

## Requirements

- Python 3.9+
- Webcam or video file
- Internet connection on first run so Ultralytics can download the YOLO model automatically

## Installation

Create and activate a virtual environment if desired.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run with webcam

```bash
python app.py --source 0
```

## Run with a video

```bash
python app.py --source sample.mp4
```

## Record annotated output

```bash
python app.py --source sample.mp4 --record
```

## Use a different config

```bash
python app.py --source 0 --config config/config.json
```

## Controls

- `q` = quit
- `s` = save a manual snapshot
- `p` = pause/resume
- `r` = reset counters

## Configuration

Edit:

`config/config.json`

Example settings include:

- YOLO model name
- confidence threshold
- maximum crowd capacity
- ROI polygon
- alert cooldown
- crowd-density ranges
- entry/exit counting line
- whether to show object IDs

Coordinates in `roi.normalized_points` are normalized from `0.0` to `1.0`.

Example:

```json
[[0.05, 0.1], [0.95, 0.1], [0.95, 0.95], [0.05, 0.95]]
```

means nearly the full image.

## Output

The system writes:

- `output/logs/events.csv`
- `output/snapshots/*.jpg`
- `output/annotated_output.mp4` when `--record` is used

## Project structure

```text
advanced_crowd_detection/
│
├── app.py
├── requirements.txt
├── README.md
├── config/
│   └── config.json
├── crowd_detection/
│   ├── __init__.py
│   ├── detector.py
│   ├── analytics.py
│   ├── drawing.py
│   ├── logger.py
│   └── utils.py
└── output/
    ├── logs/
    └── snapshots/
```

## How it works

1. YOLO detects people.
2. Ultralytics tracking assigns persistent IDs.
3. Only people whose center point falls inside the configured ROI are counted.
4. Crowd density is calculated using current occupancy versus configured capacity.
5. When occupancy reaches the alert threshold, the system:
   - displays an on-screen warning,
   - writes an event to CSV,
   - stores a snapshot,
   - applies a cooldown before creating another automatic alert.
6. A virtual line tracks directional crossings.

## Notes

The default model is `yolov8n.pt`, which is fast and suitable for many laptops.
For higher accuracy, try `yolov8s.pt` or `yolov8m.pt`, but they require more computing power.

For NVIDIA GPU acceleration, install the appropriate CUDA-enabled PyTorch build for your system.
