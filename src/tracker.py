# src/tracker.py
# ═══════════════════════════════════════════════════════════
# PERSON TRACKING MODULE
# Uses ByteTrack to assign and maintain unique IDs
# for each detected person across video frames.
# Input: sv.Detections from detector.py
# Output: sv.Detections with tracker_id assigned to each person
# ═══════════════════════════════════════════════════════════

import supervision as sv
import numpy as np
from src.config import LOST_TRACK_BUFFER, TRACK_ACTIVATION_THRESHOLD


class PersonTracker:
    """
    Tracks detected persons across video frames using ByteTrack.
    
    Assigns a persistent unique ID to each person.
    The ID stays the same even if the person is briefly hidden.
    
    Usage:
        tracker = PersonTracker()
        tracked_detections = tracker.update(detections)
        # tracked_detections.tracker_id = [1, 2, 3, ...] IDs
    """

    def __init__(self, frame_rate: int = 30):
        """
        Initialize ByteTrack tracker.
        
        Args:
            frame_rate: FPS of the video being processed
                       Used to calculate how long to keep lost tracks
        """
        # Create ByteTrack tracker from supervision library
        # track_activation_threshold: minimum confidence to START tracking someone new
        # lost_track_buffer: how many frames to remember a lost person
        #   30 frames at 30fps = 1 second before forgetting
        # minimum_matching_threshold: how similar must positions be to match
        # frame_rate: video FPS
        self.tracker = sv.ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate
        )

        # Dictionary to store each person's movement history
        # Format: {track_id: [(cx, cy), (cx, cy), ...]}
        # This lets us analyze how each person moved over time
        self.trajectories = {}

        print(f"✅ ByteTrack tracker initialized")
        print(f"   Frame rate: {frame_rate} FPS")
        print(f"   Lost track buffer: {LOST_TRACK_BUFFER} frames")

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections from current frame.
        
        This is called for EVERY frame of the video.
        ByteTrack matches detections to existing tracks (same person)
        or creates new tracks (new person entered scene).
        
        Args:
            detections: sv.Detections from PersonDetector.detect()
        
        Returns:
            sv.Detections: Same detections but now with tracker_id set
                          detections.tracker_id = array of IDs [1, 2, 3, ...]
        """
        # If no one detected this frame, return empty
        if len(detections) == 0:
            return detections

        # Update ByteTrack with current detections
        # ByteTrack figures out which detection = which person
        # and assigns/maintains track IDs
        tracked = self.tracker.update_with_detections(detections)

        # Store trajectories for each tracked person
        # This builds up their movement history over time
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                # Get this person's bounding box
                bbox = tracked.xyxy[i]
                x1, y1, x2, y2 = bbox

                # Calculate centroid (center point)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Add to their trajectory history
                # If first time seeing this ID, create new list
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []

                # Append current position
                self.trajectories[track_id].append((cx, cy))

                # Keep only last 60 positions (2 seconds at 30fps)
                # Prevents memory from growing forever
                if len(self.trajectories[track_id]) > 60:
                    self.trajectories[track_id] = \
                        self.trajectories[track_id][-60:]

        return tracked

    def get_trajectory(self, track_id: int) -> list:
        """
        Get the movement history for a specific person.
        
        Returns list of (cx, cy) positions over time.
        First item = oldest position, Last item = most recent.
        
        Args:
            track_id: The unique ID of the person
        
        Returns:
            List of (cx, cy) tuples, empty list if not found
        """
        return self.trajectories.get(track_id, [])

    def get_direction(self, track_id: int, num_points: int = 5) -> str:
        """
        Estimate the direction a person is moving.
        
        Looks at their last few positions and calculates
        which direction they are predominantly moving.
        
        Args:
            track_id: The unique ID of the person
            num_points: How many recent positions to use
        
        Returns:
            String: 'up', 'down', 'left', 'right', or 'unknown'
        """
        trajectory = self.get_trajectory(track_id)

        # Need at least num_points positions to estimate direction
        if len(trajectory) < num_points:
            return 'unknown'

        # Get the last num_points positions
        recent = trajectory[-num_points:]

        # Calculate displacement from oldest to newest position
        # dx = how much they moved horizontally
        # dy = how much they moved vertically
        dx = recent[-1][0] - recent[0][0]  # positive = moved right
        dy = recent[-1][1] - recent[0][1]  # positive = moved down

        # Determine dominant direction
        if abs(dy) > abs(dx):
            # Moved more vertically than horizontally
            return 'down' if dy > 0 else 'up'
        elif abs(dx) > abs(dy):
            # Moved more horizontally than vertically
            return 'right' if dx > 0 else 'left'
        else:
            return 'unknown'

    def reset(self):
        """
        Reset the tracker completely.
        Used when switching between videos.
        """
        self.tracker.reset()
        self.trajectories = {}
        print("🔄 Tracker reset")