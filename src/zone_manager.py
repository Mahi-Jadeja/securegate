# src/zone_manager.py
# ═══════════════════════════════════════════════════════════
# ZONE MANAGER — Fixed to use VIDEO timestamps not wall clock
#
# KEY FIX: Crossing timestamps are now based on frame position
# in the video (frame_number / fps) not time.time().
# This makes tailgating detection correct regardless of
# how fast or slow the computer processes frames.
# ═══════════════════════════════════════════════════════════

import time as time_module


class ZoneManager:
    """
    Manages a virtual gate line and detects when tracked
    persons cross it.

    Supports two line orientations:
    - "horizontal": people cross top → bottom
    - "vertical":   people cross left → right or right → left

    IMPORTANT: Uses video-time timestamps (frame/fps) not
    wall-clock time. This ensures time gaps between crossings
    reflect actual video time, not CPU processing time.
    """

    def __init__(self,
                 line_type: str = "horizontal",
                 line_position: int = 300,
                 line_start: int = 100,
                 line_end: int = 700,
                 flip_direction: bool = False,
                 line_y: int = None,
                 line_x_start: int = None,
                 line_x_end: int = None):

        # Legacy API support
        if line_y is not None:
            line_type     = "horizontal"
            line_position = line_y
            line_start    = line_x_start if line_x_start is not None else 100
            line_end      = line_x_end   if line_x_end   is not None else 700

        self.line_type      = line_type
        self.line_position  = line_position
        self.line_start     = line_start
        self.line_end       = line_end
        self.flip_direction = flip_direction

        self.prev_positions = {}
        self.crossing_log   = {}
        self.crossed_ids    = set()
        self.in_count       = 0
        self.out_count      = 0

        print(f"✅ ZoneManager initialized")
        if line_type == "horizontal":
            print(f"   Type:     HORIZONTAL line")
            print(f"   Line:     y={line_position}, "
                  f"x=[{line_start} → {line_end}]")
            print(f"   Crossing: top → bottom = IN")
        else:
            direction_label = ("right → left = IN"
                               if flip_direction
                               else "left → right = IN")
            print(f"   Type:     VERTICAL line")
            print(f"   Line:     x={line_position}, "
                  f"y=[{line_start} → {line_end}]")
            print(f"   Crossing: {direction_label}")

    def update(self, tracked_detections,
               video_timestamp: float = None) -> list:
        """
        Check all tracked persons for line crossings.

        Args:
            tracked_detections: sv.Detections with tracker_id
            video_timestamp: Current position in video in SECONDS
                            e.g. frame_number / fps
                            If None, falls back to wall clock time.
                            ALWAYS pass this for correct tailgating detection.

        Returns:
            List of new crossing dicts this frame.
        """
        new_crossings = []

        if tracked_detections is None:
            return new_crossings
        if tracked_detections.tracker_id is None:
            return new_crossings
        if len(tracked_detections) == 0:
            return new_crossings

        for i, track_id in enumerate(tracked_detections.tracker_id):
            bbox = tracked_detections.xyxy[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if self.line_type == "horizontal":
                crossing = self._check_horizontal(
                    track_id, cx, cy, video_timestamp)
            else:
                crossing = self._check_vertical(
                    track_id, cx, cy, video_timestamp)

            if crossing is not None:
                new_crossings.append(crossing)

            self.prev_positions[track_id] = (cx, cy)

        return new_crossings

    def _check_horizontal(self, track_id, cx, cy,
                           timestamp=None) -> dict:
        """Check horizontal line crossing."""
        if cx < self.line_start or cx > self.line_end:
            return None
        if track_id not in self.prev_positions:
            return None
        if track_id in self.crossed_ids:
            return None

        prev_cx, prev_cy = self.prev_positions[track_id]

        if prev_cy < self.line_position and cy >= self.line_position:
            return self._record_crossing(track_id, cx, cy, "IN", timestamp)
        if prev_cy >= self.line_position and cy < self.line_position:
            return self._record_crossing(track_id, cx, cy, "OUT", timestamp)

        return None

    def _check_vertical(self, track_id, cx, cy,
                         timestamp=None) -> dict:
        """Check vertical line crossing."""
        if cy < self.line_start or cy > self.line_end:
            return None
        if track_id not in self.prev_positions:
            return None
        if track_id in self.crossed_ids:
            return None

        prev_cx, prev_cy = self.prev_positions[track_id]

        if prev_cx < self.line_position and cx >= self.line_position:
            direction = "OUT" if self.flip_direction else "IN"
            return self._record_crossing(track_id, cx, cy, direction, timestamp)
        if prev_cx >= self.line_position and cx < self.line_position:
            direction = "IN" if self.flip_direction else "OUT"
            return self._record_crossing(track_id, cx, cy, direction, timestamp)

        return None

    def _record_crossing(self, track_id, cx, cy,
                          direction: str,
                          timestamp: float = None) -> dict:
        """
        Record a confirmed crossing event.

        Uses video-time timestamp so time gaps between crossings
        reflect actual video elapsed time, not CPU time.
        """
        crossing = {
            'track_id':  int(track_id),
            'direction': direction,
            'time':      (timestamp if timestamp is not None
                         else time_module.time()),
            'position':  (cx, cy)
        }

        self.crossing_log[int(track_id)] = crossing
        self.crossed_ids.add(track_id)

        if direction == "IN":
            self.in_count += 1
        else:
            self.out_count += 1

        print(f"🚶 Person {int(track_id)} crossed {direction} "
              f"at ({cx:.0f}, {cy:.0f}) "
              f"[video_time={timestamp:.2f}s]"
              if timestamp else
              f"🚶 Person {int(track_id)} crossed {direction} "
              f"at ({cx:.0f}, {cy:.0f})")

        return crossing

    def get_counts(self) -> dict:
        return {'in': self.in_count, 'out': self.out_count}

    def get_all_crossings(self) -> list:
        crossings = list(self.crossing_log.values())
        crossings.sort(key=lambda c: c['time'])
        return crossings

    def get_crossing_time(self, track_id: int):
        entry = self.crossing_log.get(int(track_id))
        return entry['time'] if entry else None

    def get_line_coords(self) -> tuple:
        if self.line_type == "horizontal":
            return (
                (self.line_start,    self.line_position),
                (self.line_end,      self.line_position)
            )
        else:
            return (
                (self.line_position, self.line_start),
                (self.line_position, self.line_end)
            )

    def reset(self):
        self.prev_positions = {}
        self.crossing_log   = {}
        self.crossed_ids    = set()
        self.in_count       = 0
        self.out_count      = 0
        print("🔄 ZoneManager reset")