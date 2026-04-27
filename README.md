# 🔴 SecureGate — AI-Powered Tailgating Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange?style=flat-square&logo=streamlit)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Real-Time Tailgating & Piggybacking Detection at Access-Controlled Entry Points**

*Using Spatio-Temporal Analysis on Monocular Surveillance Video*

[Live Demo](https://securegate-nsu6l5vod73akyasgoufvm.streamlit.app/) • [Report]([research_report/CV_GROUP_15.pdf](https://github.com/Mahi-Jadeja/securegate/blob/ba2f263fd54c8cd1ad51403078ec2bec037c576a/research_report/CV_GROUP_15.pdf)) • [Dataset](#dataset)

</div>

---

---

## 📌 Overview

**SecureGate** is an AI-powered security system that automatically detects two types of unauthorized physical access:

| Threat | Description | Detection Method |
|--------|-------------|-----------------|
| **Tailgating** | An unauthorized person follows closely behind an authorized person through a secured door without presenting credentials | Spatio-temporal analysis — time gap + proximity between consecutive gate crossings |
| **Piggybacking** | An unauthorized person is physically carried by an authorized person to bypass access control | Bounding box aspect ratio anomaly detection |

The system processes surveillance video in real time, tracks all persons using ByteTrack, and fires multi-channel alerts (alarm sound + email notification) when a threat is detected.

---

## 🎬 Demo

<div align="center">

| Normal Entry | Tailgating Detected | Piggybacking Detected |
|:---:|:---:|:---:|
| ✅ Green banner | 🚨 Red banner + alarm | 🚨 Purple banner + alarm |

</div>

> **Live app:** [securegate.streamlit.app](https://YOUR_APP_URL.streamlit.app)

---

## 🏗 System Architecture

```
                            Video Input
                                │
                                ▼
                    ┌─────────────────────┐
                    │    YOLOv8n          │
                    │    Detection        │
                    │ - COCO class 0      │
                    │ - Conf: 0.45–0.50   │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │    ByteTrack        │
                    │    Tracker          │
                    │ - Track IDs         │
                    │ - Persistent        │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │   Zone Manager      │
                    │ - Gate line detect  │
                    │ - Crossing events   │
                    └────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌─────────────────┐      ┌──────────────────┐
      │ Tailgate Engine │      │ Piggyback        │
      │ Spatio-Temporal │      │ Detector         │
      │ Analysis        │      │ BBox ratio ≥ 3.0 │
      └────────┬────────┘      └────────┬─────────┘
               │                        │
               └────────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Alert System       │
                 │ 🔊 Alarm sound      │
                 │ 📧 Email notif      │
                 │ 📋 Event logging    │
                 └────────┬────────────┘
                          ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 │ - Live feed         │
                 │ - Real-time stats   │
                 │ - Metrics           │
                 └─────────────────────┘
```


---

## 🔬 Detection Logic

### Tailgating Detection (Spatio-Temporal Analysis)

A tailgating event is flagged when **all three conditions** are simultaneously true:

**CONDITION 1 — TEMPORAL:**
```
τ_min < Δt(person_A, person_B) < τ_time
Default: 0.5s < gap < 5.0s
```

**CONDITION 2 — SPATIAL:**
```
euclidean_distance(pos_A, pos_B) < τ_proximity
Default: distance < 150–200px (scaled to frame width)
```

**CONDITION 3 — AUTHORIZATION:**
```
No access event registered between crossing of A and B
(First person auto-authorized; subsequent entries checked)
```

### Piggybacking Detection (Bounding Box Anomaly)

```
FOR each detection near the gate line (within 10% frame height):
  hw_ratio = bbox_height / bbox_width

  Normal standing person: hw_ratio ≈ 1.8 – 3.2
  Two stacked people:    hw_ratio ≈ 3.5 – 5.5+

  IF hw_ratio > 3.0 for ≥ 2 consecutive frames:
    → PIGGYBACKING ALERT
```

### Key Design Decisions

- **Video timestamps** (frame / fps) used for all timing — not wall clock time. This ensures correct detection regardless of CPU processing speed.
- **Detection downscaling**: Large phone videos (2160×3840) are downscaled to 640px width for YOLO inference, then annotated on the original frame for display quality.
- **Per-scenario configuration**: Each of the 13 test scenarios has individually tuned thresholds in `scenarios.json`.

---

## 📊 Test Results

Evaluated on **13 scenarios** across 3 data sources:

| Metric | Value |
|--------|-------|
| **Precision** | TBD after full evaluation |
| **Recall** | TBD after full evaluation |
| **F1-Score** | TBD after full evaluation |
| **Accuracy** | TBD after full evaluation |
| **Avg FPS (ChokePoint)** | ~28 FPS |
| **Avg FPS (Self-Recorded)** | ~19 FPS |

### Scenario Coverage

| ID | Scenario | Source | Expected | Type |
|----|----------|--------|----------|------|
| C1 | Normal Single Entry | ChokePoint | NORMAL | TN |
| C2 | Normal Exit Sequence | ChokePoint | NORMAL | TN |
| C3 | Portal 2 Structured Entry | ChokePoint | NORMAL | TN |
| C4 | Portal 2 Continuation | ChokePoint | NORMAL | TN |
| C5 | Crowded Entry Stress Test | ChokePoint | NORMAL | TN |
| C6 | Crowded Exit Stress Test | ChokePoint | NORMAL | TN |
| S2 | Both Authorized Entry | Self-Recorded | NORMAL | TN |
| S3 | Safe Sequential Entry | Self-Recorded | NORMAL | TN |
| S4 | Hesitant Tailgating | Self-Recorded | TAILGATING | TP |
| S5 | Rush Tailgating | Self-Recorded | TAILGATING | TP |
| S6 | Group Tailgating | Self-Recorded | TAILGATING | TP |
| S8 | Piggybacking Detection | Self-Recorded | PIGGYBACKING | TP |
| S13 | Five Person Sequential | Self-Recorded | NORMAL | TN |

---

## 🗂 Project Structure

```
securegate/
│
├── app.py                      # Streamlit dashboard (main entry point)
├── scenarios.json              # All 13 test scenario configurations
├── video_ids.json              # Google Drive file IDs for cloud deployment
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for Streamlit Cloud)
│
├── src/
│   ├── config.py               # Central configuration (thresholds, settings)
│   ├── detector.py             # YOLOv8n person detection wrapper
│   ├── tracker.py              # ByteTrack multi-object tracking wrapper
│   ├── zone_manager.py         # Virtual gate line + crossing detection
│   ├── tailgate_engine.py      # Core spatio-temporal tailgating logic
│   ├── piggyback_detector.py   # Bounding box ratio piggybacking detection
│   ├── alert_system.py         # Alarm sound + email notification
│   ├── video_processor.py      # Main pipeline (connects all modules)
│   ├── video_manager.py        # Cloud video download from Google Drive
│   └── __init__.py
│
├── tests/
│   ├── test_pipeline.py        # Full test runner for all 13 scenarios
│   ├── calibrate_all.py        # Interactive gate line calibration tool
│   ├── debug_piggyback.py      # Piggyback detection debugging utility
│   └── debug_portrait.py       # Portrait mode video debugging
│
├── assets/
│   ├── alarm.wav               # PCM 16-bit alarm sound (5 seconds)
│   └── generate_alarm.py       # Script to regenerate alarm sound
│
├── data/
│   ├── chokepoint/videos/      # ChokePoint dataset MP4 files
│   └── self_recorded/          # Self-recorded MOV files
│
├── results/
│   ├── screenshots/            # Annotated frame screenshots
│   └── metrics/                # Evaluation metrics output
│
├── convert_chokepoint.py       # One-time script: frames → MP4 conversion
│
└── .streamlit/
    └── config.toml             # Streamlit theme configuration
```


---

## ⚙️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Person Detection | YOLOv8n (Ultralytics) | 8.0+ |
| Multi-Object Tracking | ByteTrack (via Supervision) | 0.27+ |
| Computer Vision | OpenCV | 4.13+ |
| Web Dashboard | Streamlit | 1.28+ |
| Alert Sound | Pygame | 2.6+ |
| Language | Python | 3.11 |
| Hardware | CPU only (MacBook M4) | — |

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11+
- macOS / Linux / Windows
- ~2 GB disk space for videos and model

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/securegate.git
cd securegate

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate alarm sound
python assets/generate_alarm.py

# 5. Download ChokePoint dataset
# Visit: https://arma.sourceforge.net/chokepoint/
# Download P1E_S1, P1L_S1, P2E_S1, P2E_S5, P2L_S5 sequences
# Extract to: data/chokepoint/

# 6. Convert ChokePoint frames to video
python convert_chokepoint.py

# 7. Add self-recorded videos to data/self_recorded/
# See docs/ for recording guide

# 8. Configure email alerts (optional)
cp .env.example .env
# Edit .env with your Gmail credentials
```

### Running the App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Running Tests

```bash
# Test all 13 scenarios
python tests/test_pipeline.py

# Calibrate gate line for a specific video
python tests/calibrate_all.py

# Test single scenario
python -c "
import json
from tests.test_pipeline import test_scenario
with open('scenarios.json') as f:
    data = json.load(f)
s = next(x for x in data['scenarios'] if x['id'] == 'C1')
test_scenario(s)
"
```
## 📁 Dataset

### ChokePoint Dataset (Public)

**Reference:** Wren, C., et al. The ChokePoint Dataset: A Tool for Human Identification Research. 2011.

- **Source:** https://arma.sourceforge.net/chokepoint/
- **Content:** Controlled portal entry/exit sequences, multiple cameras
- **Used sequences:** P1E_S1_C1, P1L_S1_C1, P2E_S1_C1, P2E_S5_C1, P2L_S5_C1
- **Format:** JPG frames → converted to MP4 at 30fps
- **Resolution:** 800×600

### Self-Recorded Dataset

7 videos recorded at a controlled doorway using a smartphone:

| Video | Duration | Scenario |
|-------|----------|----------|
| edge_two_close_auth.mov | ~6s | Two authorized persons |
| normal_two_authorized.mov | ~15s | Safe sequential entry |
| tailgate_hesitant.mov | ~9s | Delayed tailgating |
| tailgate_rush.mov | ~6s | Aggressive tailgating |
| tailgate_group.mov | ~8s | Group tailgating (2 followers) |
| piggyback_carry.mov | ~8s | Person carried on back |
| four_people_sequential_entry.mov | ~20s | Multi-person sequential |

**Recording Setup:**
- Device: Smartphone mounted at 30–45° downward angle
- Simulates CCTV camera perspective
- Resolution: 1080×1920 to 2160×3840
## 🔧 Configuration

All detection parameters are configured per-scenario in `scenarios.json`:

```json
{
  "id": "S5",
  "name": "Rush Tailgating",
  "line_y": 2578,
  "line_x_start": 394,
  "line_x_end": 1728,
  "time_threshold": 5.0,
  "proximity_threshold": 200,
  "confidence": 0.45,
  "check_piggyback": false,
  "frame_skip": 3
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `line_y` | Y-coordinate of gate line in original video | calibrated per video |
| `time_threshold` | Max seconds between crossings to flag tailgating | 5.0s |
| `proximity_threshold` | Max pixel distance between persons at crossing | 150px |
| `confidence` | YOLO detection confidence threshold | 0.5 |
| `check_piggyback` | Enable piggybacking detection (S8 only) | false |
| `frame_skip` | Process every Nth frame (1=all, 3=every 3rd) | 1 |
## 📧 Email Alerts

SecureGate sends email notifications when threats are detected.

### Local Setup

Create a `.env` file in the project root:

```env
EMAIL_ENABLED=true
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_RECEIVER=security@yourorg.com
```

**Gmail App Password Setup:**

1. Enable 2-Factor Authentication on your Gmail account
2. Go to Google Account → Security → App Passwords
3. Generate a new app password
4. Use the 16-character password (not your regular password)

### Cloud Deployment

Add secrets in Streamlit Cloud dashboard:

1. Go to Settings → Secrets
2. Paste your credentials in the same `.env` format
## ⚠️ Limitations

- **No hardware integration** — The system cannot verify actual card swipe events. The first person to cross the gate is assumed authorized.
- **Rule-based thresholds** — Detection quality depends on correct threshold calibration per camera setup.
- **Single camera** — No multi-camera fusion. Blind spots are not covered.
- **Piggybacking detection** — Relies on bounding box shape. If YOLOv8 detects two stacked people as a single normal-height bounding box, piggybacking is missed.
- **CPU processing speed** — Self-recorded phone videos (2160×3840) process at ~19 FPS on MacBook CPU. Real-time deployment would benefit from GPU.

## 🔮 Future Work

- Integration with physical access control hardware (RFID card readers)
- Deep learning-based tailgating classifier (train CNN on crossing event sequences)
- Pose estimation for piggybacking detection (MediaPipe / OpenPose)
- Multi-camera fusion for complete coverage
- Automatic camera calibration using person height estimation
- Mobile app for security guard notifications
## 📄 Publication

This project was submitted as a final year B.Tech project in the Department of Artificial Intelligence & Machine Learning.

**Title:** Real-Time Tailgating Detection at Access-Controlled Entry Points Using Spatio-Temporal Analysis on Monocular Surveillance Video

**Report:** Available in `docs/` directory

## 🙏 Acknowledgements

- **Ultralytics YOLOv8** — Object detection
- **Roboflow Supervision** — ByteTrack implementation and annotation utilities
- **ChokePoint Dataset** — Public benchmark dataset
- **Streamlit** — Web dashboard framework
## 📃 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<div align="center">

**Built with ❤️ for security research**

[🔴 Live Demo](https://YOUR_APP_URL.streamlit.app) • [🐛 Report Bug](https://github.com/YOUR_USERNAME/securegate/issues) • [💬 Discuss](https://github.com/YOUR_USERNAME/securegate/discussions)

</div>
