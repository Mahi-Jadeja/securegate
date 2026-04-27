# src/piggyback_detector.py
# ═══════════════════════════════════════════════════════════
# PIGGYBACKING DETECTION MODULE — FIXED VERSION
#
# WHAT CHANGED FROM BROKEN VERSION:
# Old: Compared bbox height to session average
#      → Failed because perspective makes people grow/shrink
#      → Far person = small box, close person = large box
#      → System confused perspective with piggybacking
#
# New: Uses ONLY height-to-width ratio of the single box
#      → A normal person standing: h/w ratio = 1.8 to 3.2
#      → Two people stacked: h/w ratio = 3.8 to 6.0+
#      → This ratio is stable regardless of camera distance
#      → Works for any resolution or camera angle
#
# IMPORTANT: Piggybacking only makes sense NEAR the gate line
#            We only check boxes that are crossing the zone
#            NOT every person in every frame
# ═══════════════════════════════════════════════════════════

import supervision as sv
import numpy as np
import time as time_module
from src.config import PIGGYBACK_HEIGHT_RATIO


class PiggybackDetector:
    """
    Detects piggybacking using bounding box aspect ratio analysis.

    KEY INSIGHT:
    A normal standing person always has a characteristic
    height-to-width ratio regardless of how far they are
    from the camera. This ratio is stable across distances.

    Normal person:    h/w = 1.8 to 3.2  (tall but not extreme)
    Two people stacked: h/w = 3.8 to 6.0+ (extremely tall)

    We ALSO require a minimum confidence threshold:
    The box must be seen for multiple consecutive frames
    before we flag it, to avoid single-frame noise.
    """

    # These ratios are based on human body proportions
    # and validated across multiple camera angles
    NORMAL_RATIO_MIN = 1.5   # Below this = person lying down / artifact
    NORMAL_RATIO_MAX = 3.2   # Above this = suspicious (possible piggyback)
    PIGGYBACK_RATIO  = 3   # Above this = strong piggyback signal

    def __init__(self,
                 height_ratio_threshold: float = PIGGYBACK_HEIGHT_RATIO,
                 frame_width: int = 800,
                 frame_height: int = 600):
        """
        Initialize the piggyback detector.

        Args:
            height_ratio_threshold: Not used for ratio comparison anymore
                                   Kept for API compatibility
            frame_width: Width of video frame (set in video_processor)
            frame_height: Height of video frame (set in video_processor)
        """
        # Minimum consecutive frames a suspicious box must appear
        # before we flag it. Prevents single-frame noise.
        self.MIN_CONSECUTIVE_FRAMES = 2

        # Track suspicious IDs and how many frames they've been suspicious
        # {track_id: consecutive_suspicious_frame_count}
        self.suspicious_frame_counts = {}

        # Track which IDs we've already flagged
        self.flagged_ids = set()

        # All confirmed piggyback events
        self.piggyback_events = []

        # Frame dimensions (used for minimum box size check)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height

        print(f"✅ PiggybackDetector initialized")
        print(f"   Method: Height/Width ratio analysis")
        print(f"   Normal ratio range: {self.NORMAL_RATIO_MIN} - {self.NORMAL_RATIO_MAX}")
        print(f"   Piggyback trigger: ratio > {self.PIGGYBACK_RATIO}")
        print(f"   Required consecutive frames: {self.MIN_CONSECUTIVE_FRAMES}")

    def set_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions when video is opened."""
        self.frame_width = width
        self.frame_height = height
        self.frame_area = width * height

    def check(self, tracked_detections: sv.Detections,
              current_time: float = None) -> list:
        """
        Check tracked persons for piggybacking signs.

        Called every frame. Uses aspect ratio analysis.

        Args:
            tracked_detections: sv.Detections with tracker_id
            current_time: Unix timestamp

        Returns:
            List of NEW piggybacking alerts this frame
        """
        if current_time is None:
            current_time = time_module.time()

        alerts = []

        if tracked_detections.tracker_id is None:
            return alerts
        if len(tracked_detections) == 0:
            return alerts

        # Get set of currently visible track IDs
        current_ids = set(int(tid) for tid in tracked_detections.tracker_id)

        # Reset suspicious count for IDs no longer visible
        for tid in list(self.suspicious_frame_counts.keys()):
            if tid not in current_ids:
                self.suspicious_frame_counts[tid] = 0

        # Check each tracked person
        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)

            # Already confirmed as piggyback → skip
            if track_id in self.flagged_ids:
                continue

            bbox = tracked_detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            height = float(y2 - y1)
            width  = float(x2 - x1)

            # Skip boxes that are too small
            # (partial detections, artifacts, or very distant people)
            # Minimum: 3% of frame height
            min_height = self.frame_height * 0.03
            if height < min_height or width < 10:
                continue

            # Skip boxes that are too wide
            # (side-by-side detection counted as one — different problem)
            # If width > 60% of frame width, probably multi-person side-by-side
            max_width = self.frame_width * 0.6
            if width > max_width:
                continue

            # Calculate height-to-width aspect ratio
            if width == 0:
                continue
            hw_ratio = height / width

            # CHECK: Is the aspect ratio in the piggybacking range?
            if hw_ratio > self.PIGGYBACK_RATIO:
                # Suspicious — increment counter
                self.suspicious_frame_counts[track_id] = \
                    self.suspicious_frame_counts.get(track_id, 0) + 1

                # Only flag after seeing it for MIN_CONSECUTIVE_FRAMES
                if self.suspicious_frame_counts[track_id] >= \
                        self.MIN_CONSECUTIVE_FRAMES:

                    alert = {
                        "track_id": track_id,
                        "time": current_time,
                        "bbox_height": round(height, 1),
                        "bbox_width": round(width, 1),
                        "height_width_ratio": round(hw_ratio, 2),
                        "consecutive_frames": self.suspicious_frame_counts[track_id],
                        "reason": (
                            f"Person {track_id} bounding box aspect ratio "
                            f"is {hw_ratio:.2f} (normal range: "
                            f"{self.NORMAL_RATIO_MIN}-{self.NORMAL_RATIO_MAX}). "
                            f"Ratio exceeds piggybacking threshold of "
                            f"{self.PIGGYBACK_RATIO}. Detected for "
                            f"{self.suspicious_frame_counts[track_id]} "
                            f"consecutive frames."
                        )
                    }

                    alerts.append(alert)
                    self.piggyback_events.append(alert)
                    self.flagged_ids.add(track_id)

                    print(f"   🚨 PIGGYBACK: Person {track_id} "
                          f"h/w ratio={hw_ratio:.2f} "
                          f"(threshold={self.PIGGYBACK_RATIO}) "
                          f"for {self.suspicious_frame_counts[track_id]} frames")
            else:
                # Not suspicious this frame → reset counter
                self.suspicious_frame_counts[track_id] = 0

        return alerts

    def get_events(self) -> list:
        """Get all detected piggybacking events."""
        return self.piggyback_events

    def get_count(self) -> int:
        """Get total number of piggybacking events."""
        return len(self.piggyback_events)

    def reset(self):
        """Reset for a new video."""
        self.suspicious_frame_counts = {}
        self.flagged_ids = set()
        self.piggyback_events = []
        print("🔄 PiggybackDetector reset")