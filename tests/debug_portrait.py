# tests/debug_portrait.py
# Debug what portrait video looks like after rotation
# and where people actually are

import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import PersonDetector

video_path = "data/self_recorded/tailgate_rush.mov"

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original video: {w}x{h}, {total} frames")
print(f"Is portrait: {h > w}")

# Read frame from middle of video
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Cannot read frame")
    exit()



# Try detection at different confidence levels
for conf in [0.3, 0.4, 0.5]:
    detector = PersonDetector(confidence=conf)
    detections = detector.detect(frame)
    print(f"\nConfidence {conf}: {len(detections)} persons detected")
    if len(detections) > 0:
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = bbox
            h_box = y2 - y1
            w_box = x2 - x1
            cy = (y1 + y2) / 2
            print(f"  Person {i+1}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                  f" h={h_box:.0f} w={w_box:.0f} cy={cy:.0f}")

# Save an annotated frame to look at
frame_h, frame_w = frame.shape[:2]

# Scale to display size
scale = min(1200/frame_w, 800/frame_h)
display = cv2.resize(frame, (int(frame_w*scale), int(frame_h*scale)))
disp_h, disp_w = display.shape[:2]

# Draw grid
for y in range(0, disp_h, 100):
    cv2.line(display, (0,y), (disp_w,y), (50,50,50), 1)
    cv2.putText(display, f"y={int(y/scale)}", (3, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
for x in range(0, disp_w, 100):
    cv2.line(display, (x,0), (x,disp_h), (50,50,50), 1)
    cv2.putText(display, f"x={int(x/scale)}", (x+2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)

# Try to detect and draw boxes
detector = PersonDetector(confidence=0.3)
detections = detector.detect(frame)
for bbox in detections.xyxy:
    x1,y1,x2,y2 = [int(v*scale) for v in bbox]
    cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
    cy_orig = int(((bbox[1]+bbox[3])/2))
    print(f"  → Draw box at scaled ({x1},{y1},{x2},{y2}), cy_orig={cy_orig}")

save_path = "results/screenshots/debug_portrait_rotated.jpg"
os.makedirs("results/screenshots", exist_ok=True)
cv2.imwrite(save_path, display)
print(f"\n✅ Saved debug frame: {save_path}")
print(f"   Open this image to see where people are")
print(f"   and what Y coordinates to use for line_y")