# src/config.py
# ═══════════════════════════════════════════════════════════
# CENTRAL CONFIGURATION FILE
# All settings for the SecureGate system live here.
# Change a value here → it changes everywhere in the system.
# ═══════════════════════════════════════════════════════════

import os
from dotenv import load_dotenv

# Load the .env file so we can read EMAIL_SENDER etc.
# load_dotenv() looks for a file called .env in your project root
# and loads all the KEY=VALUE pairs as environment variables
load_dotenv()

# ───────────────────────────────────────────────
# DETECTION SETTINGS
# ───────────────────────────────────────────────

# Which YOLOv8 model to use
# yolov8n.pt = nano (smallest, fastest, good for CPU)
# This file downloads automatically first time you run
YOLO_MODEL = "yolov8n.pt"

# How confident YOLOv8 must be before counting someone as a person
# 0.5 = 50% confident. Lower = more detections but more mistakes.
# Higher = fewer detections but more accurate.
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))

# Minimum bounding box area as fraction of frame area
# Boxes smaller than this are probably noise/shadows, not people
# 0.003 = 0.3% of frame area
MIN_BOX_AREA_RATIO = 0.003

# ───────────────────────────────────────────────
# TRACKING SETTINGS
# ───────────────────────────────────────────────

# How many frames a person can disappear before losing their ID
# 30 frames at 30fps = 1 second. After 1 second of being gone,
# they get a new ID if they reappear.
LOST_TRACK_BUFFER = 60

# Minimum confidence to start tracking a new person
TRACK_ACTIVATION_THRESHOLD = 0.25

# ───────────────────────────────────────────────
# TAILGATING DETECTION THRESHOLDS
# ───────────────────────────────────────────────

# If person B crosses within this many seconds after person A
# AND they are close → flag as potential tailgating
# Default: 5 seconds
TIME_THRESHOLD = float(os.getenv("TIME_THRESHOLD", "5.0"))

# Minimum time gap between crossings
# If two people cross within 0.5 seconds they are probably
# side-by-side, not tailgating. Ignore these.
MIN_TIME_GAP = float(os.getenv("MIN_TIME_GAP", "0.5"))

# If person A and person B are within this many pixels of each other
# when crossing the line → close enough to be suspicious
# 150 pixels ≈ roughly 1 person-width apart
PROXIMITY_THRESHOLD = float(os.getenv("PROXIMITY_THRESHOLD", "150"))

# ───────────────────────────────────────────────
# PIGGYBACKING DETECTION THRESHOLD
# ───────────────────────────────────────────────

# If a detected bounding box is taller than this ratio
# compared to average person height → flag as piggyback
# 1.6 = 60% taller than normal → two people stacked
PIGGYBACK_HEIGHT_RATIO = float(os.getenv("PIGGYBACK_HEIGHT_RATIO", "1.6"))

# ───────────────────────────────────────────────
# ALERT SETTINGS
# ───────────────────────────────────────────────

# Path to alarm sound file
ALARM_SOUND_PATH = "assets/alarm.wav"

# Minimum seconds between alarms
# Prevents alarm from spamming repeatedly
# 10 seconds = alarm plays, then 10 second silence, then can play again
ALARM_COOLDOWN = float(os.getenv("ALARM_COOLDOWN", "10.0"))

# ───────────────────────────────────────────────
# EMAIL SETTINGS
# ───────────────────────────────────────────────
# In src/config.py, replace the email section with:
import os

def _get_secret(key: str, default: str = "") -> str:
    """
    Read from Streamlit secrets if available, else .env file.
    Safe to call at module import time.
    """
    # First try environment variable (works everywhere)
    env_val = os.getenv(key, "")
    if env_val:
        return env_val

    # Then try Streamlit secrets (only works inside running app)
    try:
        import streamlit as st
        # Use hasattr check to avoid errors during import
        if hasattr(st, "secrets"):
            val = st.secrets.get(key, default)
            return str(val) if val is not None else default
    except Exception:
        pass

    return default


# Read email settings safely
# These read from .env locally and Streamlit secrets on cloud
EMAIL_ENABLED  = _get_secret("EMAIL_ENABLED",  "false").lower() == "true"
EMAIL_SENDER   = _get_secret("EMAIL_SENDER",   "")
EMAIL_PASSWORD = _get_secret("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = _get_secret("EMAIL_RECEIVER", "")

# Gmail SMTP server settings (these never change for Gmail)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# ───────────────────────────────────────────────
# VIDEO PROCESSING SETTINGS
# ───────────────────────────────────────────────

# Width to resize frames for display (pixels)
# Smaller = faster processing but lower quality display
DISPLAY_WIDTH = 800

# Process every Nth frame
# 1 = process every frame (slow but accurate)
# 2 = process every 2nd frame (faster)
# Recommended: 1 for self-recorded, 2 for ChokePoint (they are long)
FRAME_SKIP = 1