# tests/debug_piggyback.py
# See what h/w ratios are detected in the piggyback video

import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import PersonDetector

video_path = "data/self_recorded/piggyback_carry.mov"
cap        = cv2.VideoCapture(video_path)
total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {orig_w}x{orig_h}, {total} frames")

detector   = PersonDetector(confidence=0.35)
det_scale  = 640 / orig_w
gate_y_det = int(2457 * det_scale)

print(f"Detection scale: {det_scale:.3f}")
print(f"Gate y (detection space): {gate_y_det}")
print(f"\nScanning frames for detections near gate...\n")

# Sample frames across the video
sample_frames = list(range(150, total, 5))

for fn in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        break

    # Downscale
    det_w     = int(orig_w * det_scale)
    det_h     = int(orig_h * det_scale)
    det_frame = cv2.resize(frame, (det_w, det_h))

    detections = detector.detect(det_frame)

    if len(detections) == 0:
        continue

    for bbox in detections.xyxy:
        x1, y1, x2, y2 = bbox
        h_box = y2 - y1
        w_box = x2 - x1
        cy    = (y1 + y2) / 2

        if w_box == 0:
            continue

        ratio = h_box / w_box
        dist_from_gate = abs(cy - gate_y_det)

        print(f"  Frame {fn:4d} | "
              f"h={h_box:6.0f} w={w_box:5.0f} | "
              f"ratio={ratio:5.2f} | "
              f"cy={cy:6.0f} | "
              f"dist_from_gate={dist_from_gate:6.0f}px | "
              f"{'⭐ NEAR GATE' if dist_from_gate < 300 else ''}")

cap.release()
print("\nDone. Look for frames where ratio is high near gate.")
print("That ratio is what piggyback_ratio_override should be set to.")