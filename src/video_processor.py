# src/video_processor.py
# ═══════════════════════════════════════════════════════════
# MAIN VIDEO PROCESSING PIPELINE
#
# FIXES IN THIS VERSION:
# 1. No rotation at all — process every video as-is
# 2. Frame dimensions passed to PiggybackDetector correctly
# 3. Proximity threshold normalized to frame width
# 4. Gate line drawing works for horizontal and vertical lines
# 5. Removed duplicate frame printing bug
# 6. Fixed NameError: proc_width → self.proc_width
# 7. NEW: Detection downscaling for fast FPS on large videos
#    Phone videos (2160x3840) now process at 640px width
#    This gives 30+ FPS instead of 9-10 FPS
#    Annotation still happens on original frame (good quality)
# ═══════════════════════════════════════════════════════════

import cv2
import numpy as np
import time
from typing import Generator

import supervision as sv

from src.detector import PersonDetector
from src.tracker import PersonTracker
from src.zone_manager import ZoneManager
from src.tailgate_engine import TailgateEngine
from src.piggyback_detector import PiggybackDetector
from src.alert_system import AlertSystem
from src.config import DISPLAY_WIDTH, FRAME_SKIP


class VideoProcessor:
    """
    Main processing pipeline for SecureGate.

    Processes a video file frame by frame through the complete
    detection pipeline and yields annotated frames for display.

    Key design decisions:
    - No rotation applied to any video
    - Detection runs on downscaled frame (fast)
    - Annotation runs on original frame (good quality)
    - Zone line coordinates stored in detection space
    - Video timestamps used (not wall clock) for tailgating
    """

    # Maximum width for detection frame
    # Larger = slower but more accurate
    # 640 = good balance for CPU processing
    DETECTION_MAX_WIDTH = 640

    def __init__(self,
                 scenario: dict,
                 time_threshold: float = None,
                 proximity_threshold: float = None,
                 confidence: float = None):
        """
        Initialize the processor with a scenario configuration.

        Args:
            scenario: Dict from scenarios.json for one scenario
            time_threshold: Override from UI slider (optional)
            proximity_threshold: Override from UI slider (optional)
            confidence: Override from UI slider (optional)
        """
        self.scenario = scenario

        # Use UI overrides if provided, else use scenario defaults
        self.time_threshold = (time_threshold
                               or scenario.get("time_threshold", 5.0))
        self.proximity_threshold_config = (proximity_threshold
                                           or scenario.get(
                                               "proximity_threshold", 150))
        self.confidence = confidence or scenario.get("confidence", 0.5)

        # Gate line coordinates from scenarios.json
        # These are in ORIGINAL video coordinate space
        # They get converted to detection space in _init_video_dependent_modules
        self.line_y       = scenario["line_y"]
        self.line_x_start = scenario["line_x_start"]
        self.line_x_end   = scenario["line_x_end"]

        # Video file path
        self.video_path = scenario["video_path"]

        # Only run piggyback check for scenarios that need it (S8 only)
        self.check_piggyback = scenario.get("check_piggyback", False)

        # ── Frame dimension tracking ───────────────────────
        # Original = actual video dimensions
        # Detection = downscaled dimensions used for YOLO
        # These are set in _init_video_dependent_modules
        self.video_width    = None
        self.video_height   = None
        self.proc_width     = None
        self.proc_height    = None
        self.det_scale      = 1.0      # scale factor original → detection
        self.det_line_y     = None     # line_y in detection space
        self.det_line_x_start = None
        self.det_line_x_end   = None
        self.video_fps      = 30       # default, updated when video opens

        self.effective_proximity = self.proximity_threshold_config

        # ── Initialize modules that don't need frame size ──
        self.detector = PersonDetector(confidence=self.confidence)
        self.tracker  = PersonTracker(frame_rate=30)

        # These need frame size → initialized later
        self.zone            = None
        self.tailgate_engine = None

        self.piggyback_detector = PiggybackDetector(
            height_ratio_threshold=scenario.get("piggyback_height_ratio", 1.6)
        )
        self.alert_system = AlertSystem()

        # ── Status display ─────────────────────────────────
        self.current_status        = "NORMAL"
        self.status_frame_counter  = 0
        self.status_display_frames = 90   # 3 seconds at 30 fps

        # ── FPS tracking ───────────────────────────────────
        self.fps_start_time  = time.time()
        self.fps_frame_count = 0
        self.current_fps     = 0.0

        # ── Frame counter ──────────────────────────────────
        self.frame_count = 0

        print(f"\n{'='*55}")
        print(f"🎥 VideoProcessor Ready")
        print(f"   Video:          {self.video_path}")
        print(f"   Gate line:      y={self.line_y}, "
              f"x=[{self.line_x_start}→{self.line_x_end}]")
        print(f"   Time threshold: {self.time_threshold}s")
        print(f"   Confidence:     {self.confidence}")
        print(f"   Piggyback check: {self.check_piggyback}")
        print(f"{'='*55}\n")

    # ══════════════════════════════════════════════════════
    # INITIALISATION
    # ══════════════════════════════════════════════════════

    def _init_video_dependent_modules(self,
                                       width: int,
                                       height: int):
        """
        Initialize all modules that require knowing frame dimensions.
        Called once immediately after the video file is opened.

        KEY CONCEPT — Two coordinate spaces:

        ORIGINAL space: actual video dimensions (e.g. 2160x3840)
            Used for: reading frames, annotating frames for display

        DETECTION space: downscaled dimensions (e.g. 360x640)
            Used for: YOLO detection, ByteTrack tracking, ZoneManager

        The line coordinates (line_y, line_x_start, line_x_end) from
        scenarios.json are in ORIGINAL space. We convert them to
        DETECTION space and store as det_line_y etc.
        The ZoneManager uses detection-space coordinates.

        When annotating, we scale bounding boxes BACK to original space.

        Args:
            width:  original video frame width in pixels
            height: original video frame height in pixels
        """
        self.video_width  = width
        self.video_height = height
        self.proc_width   = width
        self.proc_height  = height

        # ── Calculate detection downscale factor ───────────
        # If video is wider than DETECTION_MAX_WIDTH,
        # we downscale for detection to improve FPS
        if width > self.DETECTION_MAX_WIDTH:
            self.det_scale = self.DETECTION_MAX_WIDTH / width
            det_w = self.DETECTION_MAX_WIDTH
            det_h = int(height * self.det_scale)
            print(f"   🔽 Detection downscale: {width}x{height} "
                  f"→ {det_w}x{det_h} "
                  f"(scale={self.det_scale:.3f})")
            print(f"      Expected FPS improvement: "
                  f"~{int(1/self.det_scale**2)}x faster")
        else:
            self.det_scale = 1.0
            print(f"   Detection: full resolution {width}x{height}")

        # ── Convert line coords to detection space ─────────
        # Original line coords × det_scale = detection line coords
        self.det_line_y       = int(self.line_y       * self.det_scale)
        self.det_line_x_start = int(self.line_x_start * self.det_scale)
        self.det_line_x_end   = int(self.line_x_end   * self.det_scale)

        print(f"   Line (original):  y={self.line_y}, "
              f"x=[{self.line_x_start}→{self.line_x_end}]")
        print(f"   Line (detection): y={self.det_line_y}, "
              f"x=[{self.det_line_x_start}→{self.det_line_x_end}]")

        # ── Normalize proximity threshold ──────────────────
        # Use detection-space width for normalization
        det_width  = int(self.proc_width * self.det_scale)
        auto_proximity = det_width * 0.20
        self.effective_proximity = max(
            self.proximity_threshold_config,
            auto_proximity
        )
        print(f"   Proximity: {self.effective_proximity:.0f}px "
              f"(config={self.proximity_threshold_config}, "
              f"auto={auto_proximity:.0f})")

        # ── Zone Manager (uses detection-space coords) ─────
        line_type      = self.scenario.get("line_type", "horizontal")
        flip_direction = self.scenario.get("flip_direction", False)

        self.zone = ZoneManager(
            line_type=line_type,
            line_position=self.det_line_y,
            line_start=self.det_line_x_start,
            line_end=self.det_line_x_end,
            flip_direction=flip_direction
        )

        # ── Tailgate Engine ────────────────────────────────
        self.tailgate_engine = TailgateEngine(
            time_threshold=self.time_threshold,
            proximity_threshold=self.effective_proximity
        )

        # ── Piggyback Detector (uses detection-space dims) ─
        det_w = int(self.proc_width  * self.det_scale)
        det_h = int(self.proc_height * self.det_scale)
        self.piggyback_detector.set_frame_dimensions(det_w, det_h)

    # ══════════════════════════════════════════════════════
    # FRAME PREPARATION
    # ══════════════════════════════════════════════════════

    def _get_detection_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Downscale frame for detection if needed.

        For large phone videos (2160x3840):
            Detection frame: 360x640 (36x fewer pixels → much faster)
            Original frame:  2160x3840 (kept for display annotation)

        For ChokePoint videos (800x600):
            Already small enough → no downscaling needed
            Returns original frame unchanged

        Args:
            frame: Original BGR frame from OpenCV

        Returns:
            Downscaled frame for YOLO/ByteTrack processing
        """
        if self.det_scale < 1.0:
            det_w = int(frame.shape[1] * self.det_scale)
            det_h = int(frame.shape[0] * self.det_scale)
            return cv2.resize(frame, (det_w, det_h),
                              interpolation=cv2.INTER_LINEAR)
        return frame

    # ══════════════════════════════════════════════════════
    # NEAR-GATE FILTER
    # ══════════════════════════════════════════════════════

    def _filter_near_gate(self,
                           tracked: sv.Detections) -> sv.Detections:
        """
        Return only detections whose centroid is close to the gate line.

        Used before piggyback check — only makes sense to check
        for piggybacking when the person is actually AT the gate.
        Prevents false alarms from people walking far from gate.

        Margin = 10% of detection frame height, capped at 200px.
        The cap prevents large-resolution videos from having
        an overly generous margin.

        Args:
            tracked: sv.Detections with tracker_id

        Returns:
            Filtered sv.Detections (subset near gate line)
        """
        if tracked is None or len(tracked) == 0:
            return tracked
        if tracked.tracker_id is None:
            return tracked

        det_h = int(self.proc_height * self.det_scale)
        raw_margin = det_h * 0.10
        margin = min(raw_margin, 200)

        keep = []
        for i in range(len(tracked)):
            bbox = tracked.xyxy[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if self.zone.line_type == "horizontal":
                if abs(cy - self.zone.line_position) <= margin:
                    keep.append(i)
            else:
                if abs(cx - self.zone.line_position) <= margin:
                    keep.append(i)

        if not keep:
            return tracked[0:0]

        return tracked[np.array(keep)]

    # ══════════════════════════════════════════════════════
    # MAIN PROCESSING LOOP
    # ══════════════════════════════════════════════════════

    def process(self) -> Generator:
        """
        Process the video frame by frame and yield results.

        TWO-FRAME STRATEGY:
            det_frame  = downscaled frame → YOLO + ByteTrack (fast)
            orig_frame = original frame   → annotation (good quality)

        Video timestamps (frame/fps) are used for all timing,
        NOT wall-clock time. This ensures tailgating time gaps
        are accurate regardless of CPU processing speed.

        Yields one dict per processed frame with keys:
            frame, status, frame_number, total_frames, fps,
            persons_detected, crossings, tailgating_events,
            piggyback_events, alerts, latest_event,
            alert_log, is_complete
        """
        # ── Open video ─────────────────────────────────────
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return

        # ── Read video properties ──────────────────────────
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = int(cap.get(cv2.CAP_PROP_FPS))
        video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Guard against 0 fps (corrupted video metadata)
        if video_fps <= 0:
            video_fps = 30

        self.video_fps = video_fps

        print(f"📹 Video info: {video_width}x{video_height} "
              f"@ {video_fps}fps, {total_frames} frames")

        # ── Initialize frame-size-dependent modules ────────
        self._init_video_dependent_modules(video_width, video_height)

        latest_event = None

        # ══════════════════════════════════════════════════
        # FRAME LOOP
        # ══════════════════════════════════════════════════
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Skip frames if FRAME_SKIP > 1
           # NEW — reads from scenario, falls back to config
            scenario_frame_skip = self.scenario.get("frame_skip", FRAME_SKIP)
            if self.frame_count % scenario_frame_skip != 0:
                continue

            # ── FPS tracking ───────────────────────────────
            self.fps_frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_frame_count / elapsed
            if self.fps_frame_count >= 30:
                self.fps_start_time  = time.time()
                self.fps_frame_count = 0

            # ── Video timestamp ────────────────────────────
            # Use frame position in video, NOT wall clock time
            # This is critical for correct tailgating time gaps
            video_timestamp = self.frame_count / self.video_fps

            # ── Get detection frame (downscaled) ───────────
            det_frame = self._get_detection_frame(frame)

            # ── STEP 1: Detect persons ─────────────────────
            # Runs on small det_frame → fast
            detections = self.detector.detect(det_frame)

            # ── STEP 2: Track persons ──────────────────────
            # ByteTrack coordinates are in detection space
            tracked = self.tracker.update(detections)

            # ── STEP 3: Check line crossings ───────────────
            # ZoneManager uses detection-space coordinates → correct
            new_crossings = self.zone.update(
                tracked,
                video_timestamp=video_timestamp
            )

            # ── STEP 4: Check piggybacking ─────────────────
            # Only for S8 (check_piggyback=True in scenarios.json)
            if self.check_piggyback:
                near_gate = self._filter_near_gate(tracked)
                piggyback_alerts = self.piggyback_detector.check(
                    near_gate, current_time=video_timestamp
                )
                for pb_alert in piggyback_alerts:
                    logged = self.alert_system.trigger_piggyback_alert(
                        pb_alert)
                    latest_event = logged
                    self.current_status       = "PIGGYBACKING"
                    self.status_frame_counter = self.status_display_frames

            # ── STEP 5: Check tailgating ───────────────────
            for crossing in new_crossings:
                result = self.tailgate_engine.process_crossing(crossing)

                if result["is_tailgating"]:
                    event_details = {
                        "track_id":  crossing["track_id"],
                        "behind_id": result["behind_id"],
                        "time_gap":  result["time_gap"],
                        "distance":  result["distance"],
                        "reason":    result["reason"]
                    }
                    logged = self.alert_system.trigger_tailgate_alert(
                        event_details)
                    latest_event = logged
                    self.current_status       = "TAILGATING"
                    self.status_frame_counter = self.status_display_frames

            # ── STEP 6: Status display countdown ──────────
            if self.status_frame_counter > 0:
                self.status_frame_counter -= 1
            else:
                if self.current_status in ["TAILGATING", "PIGGYBACKING"]:
                    self.current_status = "NORMAL"

            # ── STEP 7: Annotate ORIGINAL frame ───────────
            # Annotation happens on the original high-quality frame
            # Bounding box coordinates are scaled up from detection space
            annotated = self._annotate_frame(frame, tracked)

            # ── STEP 8: Print progress ─────────────────────
            if self.frame_count % 30 == 0:
                print(f"   Frame {self.frame_count}/{total_frames} "
                      f"| Status: {self.current_status} "
                      f"| People: {len(tracked) if tracked else 0} "
                      f"| FPS: {self.current_fps:.1f}")

            # ── STEP 9: Yield result ───────────────────────
            stats       = self.tailgate_engine.get_stats()
            zone_counts = self.zone.get_counts()

            yield {
                "frame":             annotated,
                "status":            self.current_status,
                "frame_number":      self.frame_count,
                "total_frames":      total_frames,
                "fps":               round(self.current_fps, 1),
                "persons_detected":  len(tracked) if tracked else 0,
                "crossings":         zone_counts["in"],
                "tailgating_events": stats["total_tailgating"],
                "piggyback_events":  self.piggyback_detector.get_count(),
                "alerts":            self.alert_system.get_alert_count(),
                "latest_event":      latest_event,
                "alert_log":         self.alert_system.get_alert_log(),
                "is_complete":       False
            }

        # ══════════════════════════════════════════════════
        # VIDEO FINISHED
        # ══════════════════════════════════════════════════
        cap.release()

        final_stats = self.tailgate_engine.get_stats()
        zone_counts = self.zone.get_counts()

        print(f"\n✅ Processing complete: {self.frame_count} frames")
        print(f"   Total crossings:   {zone_counts['in']}")
        print(f"   Tailgating events: {final_stats['total_tailgating']}")
        print(f"   Piggyback events:  {self.piggyback_detector.get_count()}")
        print(f"   Alerts fired:      {self.alert_system.get_alert_count()}")
        print(f"   Average FPS:       {self.current_fps:.1f}")

        yield {
            "frame":             None,
            "status":            self.current_status,
            "frame_number":      self.frame_count,
            "total_frames":      total_frames,
            "fps":               round(self.current_fps, 1),
            "persons_detected":  0,
            "crossings":         zone_counts["in"],
            "tailgating_events": final_stats["total_tailgating"],
            "piggyback_events":  self.piggyback_detector.get_count(),
            "alerts":            self.alert_system.get_alert_count(),
            "latest_event":      latest_event,
            "alert_log":         self.alert_system.get_alert_log(),
            "is_complete":       True
        }

    # ══════════════════════════════════════════════════════
    # FRAME ANNOTATION
    # ══════════════════════════════════════════════════════

    def _annotate_frame(self,
                         frame: np.ndarray,
                         tracked: sv.Detections) -> np.ndarray:
        """
        Draw all visual annotations on the ORIGINAL quality frame.

        Bounding box coordinates come from ByteTrack which operates
        in detection space (downscaled). We scale them back up to
        original space before drawing.

        Scale chain:
            detection coords × (1/det_scale) = original coords
            original coords  × display_scale = display coords
            Combined: detection coords × (display_scale / det_scale)

        Draws:
            Green boxes  = normal persons
            Red boxes    = tailgaters
            Purple boxes = piggyback persons
            Dashed red line = access gate line
            Status banner   = top left
            Stats overlay   = top right
        """
        annotated = frame.copy()

        # ── Resize for display ─────────────────────────────
        orig_h, orig_w = annotated.shape[:2]
        if orig_w > DISPLAY_WIDTH:
            disp_scale = DISPLAY_WIDTH / orig_w
            new_w = int(orig_w * disp_scale)
            new_h = int(orig_h * disp_scale)
            annotated = cv2.resize(annotated, (new_w, new_h))
        else:
            disp_scale = 1.0

        frame_h, frame_w = annotated.shape[:2]

        # ── Coordinate scale factors ───────────────────────
        # Detection coords → display coords
        # = go from detection space → original space → display space
        # = (1 / det_scale) × disp_scale
        # = disp_scale / det_scale
        if self.det_scale > 0:
            coord_scale_x = disp_scale / self.det_scale
            coord_scale_y = disp_scale / self.det_scale
        else:
            coord_scale_x = disp_scale
            coord_scale_y = disp_scale

        # ── Draw gate line ─────────────────────────────────
        # Line coords from ZoneManager are in detection space
        # Scale to display space for drawing
        line_coords = self.zone.get_line_coords()
        pt1 = (int(line_coords[0][0] * coord_scale_x),
                int(line_coords[0][1] * coord_scale_y))
        pt2 = (int(line_coords[1][0] * coord_scale_x),
                int(line_coords[1][1] * coord_scale_y))

        # Draw dashed line
        total_len = int(
            ((pt2[0] - pt1[0]) ** 2 +
             (pt2[1] - pt1[1]) ** 2) ** 0.5
        )
        num_segments = max(total_len // 30, 1)

        for seg in range(num_segments):
            if seg % 2 == 0:
                t1  = seg       / num_segments
                t2  = (seg + 1) / num_segments
                dp1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                       int(pt1[1] + t1 * (pt2[1] - pt1[1])))
                dp2 = (int(pt1[0] + t2 * (pt2[0] - pt1[0])),
                       int(pt1[1] + t2 * (pt2[1] - pt1[1])))
                cv2.line(annotated, dp1, dp2, (0, 0, 255), 2)

        cv2.putText(
            annotated, "ACCESS LINE",
            (pt1[0] + 5, max(pt1[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 255), 1, cv2.LINE_AA
        )

        # ── Draw bounding boxes and labels ─────────────────
        if tracked is not None and len(tracked) > 0:

            tailgate_ids = set()
            if self.tailgate_engine:
                for ev in self.tailgate_engine.tailgate_events:
                    tailgate_ids.add(ev.get("track_id"))

            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]

                # Scale from detection space to display space
                x1 = int(bbox[0] * coord_scale_x)
                y1 = int(bbox[1] * coord_scale_y)
                x2 = int(bbox[2] * coord_scale_x)
                y2 = int(bbox[3] * coord_scale_y)

                track_id = None
                if tracked.tracker_id is not None:
                    track_id = int(tracked.tracker_id[i])

                # Color by detection type
                if track_id in tailgate_ids:
                    color = (0, 0, 255)      # Red = tailgater
                elif track_id in self.piggyback_detector.flagged_ids:
                    color = (128, 0, 128)    # Purple = piggyback
                else:
                    color = (0, 255, 0)      # Green = normal

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if track_id is not None:
                    label = f"ID:{track_id}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - th - 6),
                        (x1 + tw + 4, y1),
                        color, -1
                    )
                    cv2.putText(
                        annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA
                    )

        # ── Status banner ──────────────────────────────────
        if self.current_status == "TAILGATING":
            bcolor = (0, 0, 200)
            btext  = "!! TAILGATING DETECTED!"
        elif self.current_status == "PIGGYBACKING":
            bcolor = (128, 0, 128)
            btext  = "!! PIGGYBACKING DETECTED!"
        else:
            bcolor = (0, 150, 0)
            btext  = "OK  SYSTEM NORMAL"

        cv2.rectangle(annotated, (0, 0), (370, 35), bcolor, -1)
        cv2.putText(
            annotated, btext, (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        # ── Stats overlay ──────────────────────────────────
        zone_counts = self.zone.get_counts() if self.zone else {"in": 0}
        stats_items = [
            f"People:    {len(tracked) if tracked else 0}",
            f"FPS:       {self.current_fps:.1f}",
            f"Crossings: {zone_counts['in']}",
            f"Alerts:    {self.alert_system.get_alert_count()}"
        ]
        cv2.rectangle(
            annotated,
            (frame_w - 165, 0),
            (frame_w, 90),
            (20, 20, 20), -1
        )
        for idx, item in enumerate(stats_items):
            cv2.putText(
                annotated, item,
                (frame_w - 160, 18 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA
            )

        return annotated

    # ══════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════

    def get_final_results(self) -> dict:
        """
        Get complete results after video processing is done.

        Returns:
            dict with all statistics and event logs
        """
        stats       = (self.tailgate_engine.get_stats()
                       if self.tailgate_engine else {})
        zone_counts = (self.zone.get_counts()
                       if self.zone else {"in": 0, "out": 0})

        return {
            "scenario_id":        self.scenario.get("id"),
            "scenario_name":      self.scenario.get("name"),
            "expected_result":    self.scenario.get("expected_result"),
            "total_frames":       self.frame_count,
            "total_crossings_in": zone_counts["in"],
            "authorized_entries": stats.get("authorized_count", 0),
            "tailgating_events":  stats.get("total_tailgating", 0),
            "piggyback_events":   self.piggyback_detector.get_count(),
            "total_alerts":       self.alert_system.get_alert_count(),
            "alert_log":          self.alert_system.get_alert_log(),
            "avg_fps":            round(self.current_fps, 1)
        }