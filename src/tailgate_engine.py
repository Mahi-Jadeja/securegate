# src/tailgate_engine.py
# ═══════════════════════════════════════════════════════════
# TAILGATING DETECTION ENGINE
# The core academic contribution of this project.
# Analyzes crossing events using spatio-temporal analysis
# to determine if tailgating has occurred.
#
# INPUT:  Crossing events from zone_manager.py
# OUTPUT: TailgateEvent objects when tailgating is detected
# ═══════════════════════════════════════════════════════════

import time
import math
from src.config import (
    TIME_THRESHOLD,
    MIN_TIME_GAP,
    PROXIMITY_THRESHOLD
)


class TailgateEngine:
    """
    Detects tailgating using spatio-temporal analysis.

    For each new crossing event, it checks:
    1. TEMPORAL: Was there a recent crossing? (within TIME_THRESHOLD)
    2. SPATIAL:  Were they physically close? (within PROXIMITY_THRESHOLD)
    3. AUTH:     Was there an access event in between? (authorization check)

    If ALL 3 conditions are met → TAILGATING detected.

    Usage:
        engine = TailgateEngine()
        result = engine.process_crossing(crossing_event)
        if result["is_tailgating"]:
            # Alert!
    """

    def __init__(self,
                 time_threshold: float = TIME_THRESHOLD,
                 proximity_threshold: float = PROXIMITY_THRESHOLD,
                 min_time_gap: float = MIN_TIME_GAP):
        """
        Initialize the tailgate engine with detection thresholds.

        Args:
            time_threshold: Max seconds between crossings to flag
                          Default: 5.0 seconds
            proximity_threshold: Max pixel distance between persons
                               Default: 150 pixels
            min_time_gap: Min seconds between crossings
                         Default: 0.5 (ignore same-frame crossings)
        """
        self.time_threshold = time_threshold
        self.proximity_threshold = proximity_threshold
        self.min_time_gap = min_time_gap

        # List of all crossing events recorded so far
        # Each item: {"track_id", "time", "position", "is_tailgating"}
        self.crossing_history = []

        # List of recorded access authorization events
        # Each item: float (Unix timestamp of when access was granted)
        # In simulation: first person's crossing auto-registers access
        self.access_events = []

        # List of confirmed tailgating events
        # Each item: full tailgate event dict
        self.tailgate_events = []

        # Set of track IDs confirmed as authorized
        self.authorized_ids = set()

        # Counter for display
        self.total_crossings = 0
        self.total_tailgating = 0

        print(f"✅ TailgateEngine initialized")
        print(f"   Time threshold:      {self.time_threshold}s")
        print(f"   Proximity threshold: {self.proximity_threshold}px")
        print(f"   Min time gap:        {self.min_time_gap}s")

    def register_access_event(self, timestamp: float = None):
        """
        Register an access authorization event.

        In a real system: called when someone swipes their access card.
        In our simulation: called automatically for the FIRST person
                          who crosses the line (they are assumed authorized).

        Args:
            timestamp: When the access happened (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
        self.access_events.append(timestamp)
        print(f"   🔑 Access event registered at {timestamp:.2f}")

    def process_crossing(self, crossing_event: dict) -> dict:
        """
        Process a new crossing event and determine if it is tailgating.

        This is the MAIN METHOD. Called every time someone crosses the gate line.

        Args:
            crossing_event: Dict from zone_manager with keys:
                          - "track_id": int (who crossed)
                          - "time": float (when they crossed)
                          - "position": (cx, cy) (where they crossed)
                          - "direction": "IN" or "OUT"

        Returns:
            Dict with analysis result:
            {
                "is_tailgating": bool,
                "is_first_entry": bool,
                "behind_id": int or None,
                "time_gap": float or None,
                "distance": float or None,
                "reason": str (explanation of decision)
            }
        """
        track_id = crossing_event["track_id"]
        cross_time = crossing_event["time"]
        cross_position = crossing_event["position"]
        direction = crossing_event.get("direction", "IN")

        self.total_crossings += 1

        # ──────────────────────────────────────────────────
        # CASE 1: This is the FIRST person to ever cross
        # ──────────────────────────────────────────────────
        if len(self.crossing_history) == 0:
            # First person is ALWAYS authorized by default
            # In a real system, this is where card swipe happens
            self.authorized_ids.add(track_id)
            self.register_access_event(cross_time)

            # Record this crossing
            self.crossing_history.append({
                "track_id": track_id,
                "time": cross_time,
                "position": cross_position,
                "is_tailgating": False,
                "direction": direction
            })

            result = {
                "is_tailgating": False,
                "is_first_entry": True,
                "behind_id": None,
                "time_gap": None,
                "distance": None,
                "reason": f"Person {track_id} is first entry — authorized by default"
            }
            print(f"   ✅ Person {track_id}: First entry — authorized")
            return result

        # ──────────────────────────────────────────────────
        # CASE 2: Check if we've seen this person cross before
        # (same person crossing again — ignore to avoid double-counting)
        # ──────────────────────────────────────────────────
        already_crossed_ids = [c["track_id"] for c in self.crossing_history]
        if track_id in already_crossed_ids:
            result = {
                "is_tailgating": False,
                "is_first_entry": False,
                "behind_id": None,
                "time_gap": None,
                "distance": None,
                "reason": f"Person {track_id} already logged — skipping duplicate"
            }
            return result

        # ──────────────────────────────────────────────────
        # CASE 3: Subsequent person — run spatio-temporal analysis
        # ──────────────────────────────────────────────────
        result = self._spatio_temporal_analysis(
            track_id, cross_time, cross_position, direction
        )

        # Record this crossing with its result
        self.crossing_history.append({
            "track_id": track_id,
            "time": cross_time,
            "position": cross_position,
            "is_tailgating": result["is_tailgating"],
            "direction": direction
        })

        if result["is_tailgating"]:
            self.total_tailgating += 1
            self.tailgate_events.append({
                **crossing_event,
                **result
            })
            print(f"   🚨 TAILGATING: Person {track_id} "
                  f"followed Person {result['behind_id']} "
                  f"(gap: {result['time_gap']:.1f}s, "
                  f"dist: {result['distance']:.0f}px)")
        else:
            self.authorized_ids.add(track_id)
            self.register_access_event(cross_time)
            print(f"   ✅ Person {track_id}: Normal entry — {result['reason']}")

        return result

    def _spatio_temporal_analysis(self,
                                   track_id: int,
                                   cross_time: float,
                                   cross_position: tuple,
                                   direction: str) -> dict:
        """
        The core spatio-temporal analysis algorithm.

        Checks the 3 conditions against all recent crossings.

        Args:
            track_id: ID of the person being analyzed
            cross_time: When they crossed
            cross_position: Where they crossed (cx, cy)
            direction: "IN" or "OUT"

        Returns:
            Result dict (same format as process_crossing return)
        """
        # Check against all previous crossings, newest first
        for prev_crossing in reversed(self.crossing_history):
            prev_id = prev_crossing["track_id"]
            prev_time = prev_crossing["time"]
            prev_pos = prev_crossing["position"]

            # Skip: comparing person to themselves
            if prev_id == track_id:
                continue

            # Calculate time gap between crossings
            time_gap = cross_time - prev_time

            # STOP CHECKING: If this crossing is too old
            # No point checking crossings from 30 seconds ago
            if time_gap > self.time_threshold * 3:
                break

            # ── CONDITION 1: TEMPORAL CHECK ──────────────────
            # Is the time gap suspicious?
            # Too small (< 0.5s): probably side-by-side, not following
            # Too large (> 5s): too much time passed, probably independent
            # Just right (0.5s to 5s): suspicious timing
            if not (self.min_time_gap < time_gap < self.time_threshold):
                # This previous crossing is outside time window
                # Continue checking older crossings
                continue

            # ── CONDITION 2: SPATIAL CHECK ───────────────────
            # Are they physically close to each other?
            distance = self._euclidean_distance(cross_position, prev_pos)

            if distance >= self.proximity_threshold:
                # They're far apart — probably independent entries
                continue

            # ── CONDITION 3: AUTHORIZATION CHECK ─────────────
            # Was there a new access event between these two crossings?
            # If yes → person B showed their own credentials → safe
            # If no → person B has no authorization → TAILGATING
            access_between = self._check_access_between(prev_time, cross_time)

            if access_between:
                # There WAS an access event between them
                # Person B is independently authorized
                return {
                    "is_tailgating": False,
                    "is_first_entry": False,
                    "behind_id": prev_id,
                    "time_gap": round(time_gap, 2),
                    "distance": round(distance, 2),
                    "reason": (f"Access event found between Person {prev_id} "
                              f"and Person {track_id} — both authorized")
                }

            # ── ALL 3 CONDITIONS MET → TAILGATING ────────────
            return {
                "is_tailgating": True,
                "is_first_entry": False,
                "behind_id": prev_id,
                "time_gap": round(time_gap, 2),
                "distance": round(distance, 2),
                "reason": (f"Person {track_id} crossed {time_gap:.1f}s after "
                          f"Person {prev_id}, distance {distance:.0f}px, "
                          f"no access event between them")
            }

        # No suspicious crossing found in history
        # This person is authorized (came independently)
        return {
            "is_tailgating": False,
            "is_first_entry": False,
            "behind_id": None,
            "time_gap": None,
            "distance": None,
            "reason": "No recent suspicious crossing found — authorized entry"
        }

    def _euclidean_distance(self, pos1: tuple, pos2: tuple) -> float:
        """
        Calculate straight-line distance between two points.

        Uses the Pythagorean theorem:
        distance = sqrt((x2-x1)² + (y2-y1)²)

        Args:
            pos1: (x, y) first point
            pos2: (x, y) second point

        Returns:
            float: distance in pixels
        """
        dx = pos1[0] - pos2[0]  # horizontal difference
        dy = pos1[1] - pos2[1]  # vertical difference
        return math.sqrt(dx * dx + dy * dy)

    def _check_access_between(self,
                               time1: float,
                               time2: float) -> bool:
        """
        Check if any access event occurred between two timestamps.

        Args:
            time1: Start time (earlier crossing)
            time2: End time (later crossing)

        Returns:
            True if an access event exists between time1 and time2
            False if no access event in that window
        """
        for event_time in self.access_events:
            if time1 < event_time < time2:
                return True
        return False

    def get_stats(self) -> dict:
        """
        Get current detection statistics.

        Returns:
            Dict with all current counts and events
        """
        return {
            "total_crossings": self.total_crossings,
            "authorized_count": len(self.authorized_ids),
            "total_tailgating": self.total_tailgating,
            "tailgate_events": self.tailgate_events,
            "crossing_history": self.crossing_history
        }

    def reset(self):
        """Reset all state for a new video."""
        self.crossing_history = []
        self.access_events = []
        self.tailgate_events = []
        self.authorized_ids = set()
        self.total_crossings = 0
        self.total_tailgating = 0
        print("🔄 TailgateEngine reset")