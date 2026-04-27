# tests/calibrate_all.py
# Calibration tool for ALL videos — horizontal and vertical lines
# Run: python tests/calibrate_all.py

import cv2
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_sample_frame(video_path: str) -> tuple:
    """
    Read a frame from 1/3 into the video.
    Returns (frame, original_width, original_height)
    Does NOT rotate — shows video as-is.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 3)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, 0, 0

    h, w = frame.shape[:2]
    return frame, w, h


def resize_for_display(frame: np.ndarray,
                       max_w: int = 1000,
                       max_h: int = 700) -> tuple:
    """
    Resize frame to fit on screen.
    Returns (resized_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame, scale


def calibrate_video(scenario: dict) -> dict:
    """
    Interactive calibration for one scenario.
    Supports both horizontal and vertical gate lines.
    """
    video_path = scenario["video_path"]
    line_type = scenario.get("line_type", "horizontal")

    frame, orig_w, orig_h = get_sample_frame(video_path)
    if frame is None:
        print(f"   ❌ Cannot open: {video_path}")
        return scenario

    # Resize for display
    display_frame, scale = resize_for_display(frame)
    disp_h, disp_w = display_frame.shape[:2]

    # Current line values (in DISPLAY coordinates)
    # We'll convert back to original coords on save
    if line_type == "horizontal":
        # line_y is the fixed Y, line_x_start/end are the X range
        cur_pos   = int(scenario.get("line_y", orig_h // 2) * scale)
        cur_start = int(scenario.get("line_x_start", int(orig_w * 0.1)) * scale)
        cur_end   = int(scenario.get("line_x_end",   int(orig_w * 0.9)) * scale)
    else:
        # line_x is the fixed X, line_y_start/end are the Y range
        # For vertical lines, line_y in scenarios.json = line_x position
        cur_pos   = int(scenario.get("line_y", orig_w // 2) * scale)
        cur_start = int(scenario.get("line_x_start", int(orig_h * 0.1)) * scale)
        cur_end   = int(scenario.get("line_x_end",   int(orig_h * 0.9)) * scale)

    step = 5

    print(f"\n{'─'*60}")
    print(f"📹 [{scenario['id']}] {scenario['name']}")
    print(f"   File:     {video_path}")
    print(f"   Size:     {orig_w}x{orig_h}")
    print(f"   Type:     {line_type.upper()} line")
    print(f"   Expected: {scenario['expected_result']}")
    print(f"{'─'*60}")

    if line_type == "horizontal":
        print(f"   W/S or ↑/↓ = move line UP/DOWN")
        print(f"   A/D = shrink/grow left edge")
        print(f"   Z/X = shrink/grow right edge")
    else:
        print(f"   A/D or ←/→ = move line LEFT/RIGHT")
        print(f"   W/S = shrink/grow top edge")
        print(f"   Z/X = shrink/grow bottom edge")

    print(f"   Q / Enter = SAVE and next video")
    print(f"   SPACE     = SKIP (keep current values)")
    print(f"{'─'*60}")

    while True:
        display = display_frame.copy()

        # Draw grid
        for y in range(0, disp_h, 80):
            cv2.line(display, (0, y), (disp_w, y), (50, 50, 50), 1)
            cv2.putText(display, str(int(y / scale)),
                        (3, y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (80, 80, 80), 1)
        for x in range(0, disp_w, 80):
            cv2.line(display, (x, 0), (x, disp_h), (50, 50, 50), 1)
            cv2.putText(display, str(int(x / scale)),
                        (x + 2, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (80, 80, 80), 1)

        # Draw the gate line
        if line_type == "horizontal":
            pt1 = (cur_start, cur_pos)
            pt2 = (cur_end,   cur_pos)
            info = (f"HORIZONTAL: line_y={int(cur_pos/scale)}  "
                    f"x=[{int(cur_start/scale)}→{int(cur_end/scale)}]")
        else:
            pt1 = (cur_pos, cur_start)
            pt2 = (cur_pos, cur_end)
            info = (f"VERTICAL: line_x={int(cur_pos/scale)}  "
                    f"y=[{int(cur_start/scale)}→{int(cur_end/scale)}]")

        # Draw thick red line
        cv2.line(display, pt1, pt2, (0, 0, 255), 3)
        # Draw endpoint circles
        cv2.circle(display, pt1, 7, (0, 255, 255), -1)
        cv2.circle(display, pt2, 7, (0, 255, 255), -1)

        # Info bar at bottom
        bar_y = disp_h - 55
        cv2.rectangle(display, (0, bar_y), (disp_w, disp_h),
                      (0, 0, 0), -1)
        cv2.putText(display,
                    f"[{scenario['id']}] {scenario['name']}  |  "
                    f"Expected: {scenario['expected_result']}",
                    (5, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(display, info,
                    (5, bar_y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display,
                    "Q=Save  SPACE=Skip  W/S/A/D/Z/X=Move",
                    (5, bar_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Gate Line Calibration", display)
        key = cv2.waitKey(0) & 0xFF

        # ── KEY HANDLING ───────────────────────────────────
        if key in (ord('q'), 13):        # Q or Enter → save
            break

        elif key == ord(' '):            # Space → skip
            print(f"   ⏭️  Skipped")
            cv2.destroyAllWindows()
            return scenario

        elif line_type == "horizontal":
            # Move the line position (Y)
            if key in (ord('w'), 82):    # UP
                cur_pos = max(0, cur_pos - step)
            elif key in (ord('s'), 84):  # DOWN
                cur_pos = min(disp_h - 1, cur_pos + step)
            # Adjust start (left edge)
            elif key == ord('a'):
                cur_start = max(0, cur_start - step)
            elif key == ord('d'):
                cur_start = min(cur_end - 50, cur_start + step)
            # Adjust end (right edge)
            elif key == ord('z'):
                cur_end = max(cur_start + 50, cur_end - step)
            elif key == ord('x'):
                cur_end = min(disp_w - 1, cur_end + step)

        else:  # vertical
            # Move the line position (X)
            if key in (ord('a'), 81):    # LEFT
                cur_pos = max(0, cur_pos - step)
            elif key in (ord('d'), 83):  # RIGHT
                cur_pos = min(disp_w - 1, cur_pos + step)
            # Adjust start (top edge)
            elif key == ord('w'):
                cur_start = max(0, cur_start - step)
            elif key == ord('s'):
                cur_start = min(cur_end - 50, cur_start + step)
            # Adjust end (bottom edge)
            elif key == ord('z'):
                cur_end = max(cur_start + 50, cur_end - step)
            elif key == ord('x'):
                cur_end = min(disp_h - 1, cur_end + step)

    cv2.destroyAllWindows()

    # Convert display coordinates back to original video coordinates
    updated = scenario.copy()

    if line_type == "horizontal":
        updated["line_y"]       = int(cur_pos   / scale)
        updated["line_x_start"] = int(cur_start / scale)
        updated["line_x_end"]   = int(cur_end   / scale)
        print(f"   ✅ Saved: line_y={updated['line_y']}, "
              f"x=[{updated['line_x_start']}→{updated['line_x_end']}]")
    else:
        # For vertical line: store line_x as line_y field
        # (reusing the field with different semantic meaning)
        updated["line_y"]       = int(cur_pos   / scale)
        updated["line_x_start"] = int(cur_start / scale)
        updated["line_x_end"]   = int(cur_end   / scale)
        print(f"   ✅ Saved: line_x={updated['line_y']}, "
              f"y=[{updated['line_x_start']}→{updated['line_x_end']}]")

    return updated


def main():
    with open("scenarios.json", "r") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    print(f"\n{'='*60}")
    print(f"🎯 GATE LINE CALIBRATION TOOL")
    print(f"   {len(scenarios)} scenarios to calibrate")
    print(f"{'='*60}")
    print(f"\nFor each video:")
    print(f"  HORIZONTAL line → position where people walk through")
    print(f"  VERTICAL line   → position where people cross left-right")
    print(f"\nThe RED LINE is your gate. People MUST cross it.")
    print(f"Place it at the doorway/chokepoint.\n")

    updated_scenarios = []

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}]", end=" ")

        if not os.path.exists(scenario["video_path"]):
            print(f"⚠️  NOT FOUND: {scenario['video_path']}")
            updated_scenarios.append(scenario)
            continue

        updated = calibrate_video(scenario)
        updated_scenarios.append(updated)

        # Save after every video
        data["scenarios"] = (updated_scenarios +
                             scenarios[len(updated_scenarios):])
        with open("scenarios.json", "w") as f:
            json.dump(data, f, indent=2)

    data["scenarios"] = updated_scenarios
    with open("scenarios.json", "w") as f:
        json.dump(data, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"✅ CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'ID':<5} {'Type':<12} {'Position':<10} "
          f"{'Start':<8} {'End':<8} {'Name'}")
    print(f"{'─'*60}")
    for s in updated_scenarios:
        t = s.get("line_type", "horizontal")
        label = "line_y" if t == "horizontal" else "line_x"
        print(f"{s['id']:<5} {t:<12} "
              f"{label}={s['line_y']:<6} "
              f"{s['line_x_start']:<8} {s['line_x_end']:<8} "
              f"{s['name'][:25]}")


if __name__ == "__main__":
    main()