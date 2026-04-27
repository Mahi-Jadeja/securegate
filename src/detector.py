# src/detector.py
# ═══════════════════════════════════════════════════════════
# PERSON DETECTION MODULE
# Uses YOLOv8n to find all persons in a video frame.
# Input: one video frame (an image as numpy array)
# Output: supervision Detections object with bounding boxes
# ═══════════════════════════════════════════════════════════

from ultralytics import YOLO   # YOLOv8 library
import supervision as sv        # Supervision library (helper tools)
import numpy as np              # NumPy for array operations
from src.config import (
    YOLO_MODEL,
    DETECTION_CONFIDENCE,
    MIN_BOX_AREA_RATIO
)


class PersonDetector:
    """
    Detects persons in video frames using YOLOv8n.
    
    Usage:
        detector = PersonDetector()
        detections = detector.detect(frame)
    """

    def __init__(self, model_path: str = YOLO_MODEL,
                 confidence: float = DETECTION_CONFIDENCE):
        """
        Initialize the detector by loading the YOLOv8 model.
        
        This runs ONCE when you create a PersonDetector object.
        The model file downloads automatically on first run (~6MB).
        
        Args:
            model_path: Which YOLOv8 model file to use
            confidence: Minimum confidence score (0.0 to 1.0)
        """
        # Store confidence threshold
        # We use this to filter out uncertain detections
        self.confidence = confidence

        # Load the YOLOv8 model
        # YOLO("yolov8n.pt") downloads the model if not present
        # Then loads it into memory
        print(f"⏳ Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        print(f"✅ YOLOv8 model loaded successfully")
        print(f"   Confidence threshold: {self.confidence}")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect all persons in a single video frame.
        
        This is called for EVERY frame of the video.
        
        Args:
            frame: A single video frame as a numpy array
                   Shape: (height, width, 3) - height x width x RGB channels
                   Example: (480, 640, 3) for a 640x480 video
        
        Returns:
            sv.Detections: Object containing:
                - xyxy: array of bounding boxes [[x1,y1,x2,y2], ...]
                - confidence: array of confidence scores [0.92, 0.87, ...]
                - class_id: array of class IDs [0, 0, ...] (0 = person)
                - tracker_id: None (tracker.py adds this later)
        """
        # Get frame dimensions for area filtering
        # frame.shape returns (height, width, channels)
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        # Run YOLOv8 on the frame
        # verbose=False → don't print detection results every frame
        # conf=self.confidence → only return detections above threshold
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False  # Suppress per-frame console output
        )[0]  # [0] because model returns a list, we want first item

        # Convert YOLOv8 results to supervision Detections format
        # sv.Detections is a standardized format that works with
        # ByteTrack and all other supervision tools
        detections = sv.Detections.from_ultralytics(results)

        # FILTER 1: Keep only "person" class
        # COCO dataset has 80 classes. Class 0 = person.
        # YOLOv8 might also detect chairs, cars, bags etc.
        # We only care about people.
        if len(detections) > 0:
            # detections.class_id is an array like [0, 0, 1, 0, 5]
            # class_id == 0 gives [True, True, False, True, False]
            # detections[mask] keeps only True entries
            person_mask = detections.class_id == 0
            detections = detections[person_mask]

        # FILTER 2: Remove tiny boxes (noise/shadows/false detections)
        # A box smaller than 0.3% of frame area is probably not a real person
        if len(detections) > 0:
            # xyxy is [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            # Calculate width and height of each box
            boxes = detections.xyxy
            widths = boxes[:, 2] - boxes[:, 0]   # x2 - x1
            heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
            areas = widths * heights               # width × height

            # Create mask: True if box is large enough
            min_area = frame_area * MIN_BOX_AREA_RATIO
            area_mask = areas > min_area
            detections = detections[area_mask]

        return detections

    def get_centroids(self, detections: sv.Detections) -> list:
        """
        Calculate the center point (centroid) of each bounding box.
        
        The centroid is used for:
        - Line crossing detection (is center above/below line?)
        - Distance calculation between persons
        
        Args:
            detections: sv.Detections with bounding boxes
        
        Returns:
            List of (cx, cy) tuples - center x, center y for each person
        
        Example:
            Box: x1=100, y1=50, x2=200, y2=300
            Centroid: cx = (100+200)/2 = 150, cy = (50+300)/2 = 175
        """
        centroids = []
        if len(detections) == 0:
            return centroids

        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2  # Center X
            cy = (y1 + y2) / 2  # Center Y
            centroids.append((cx, cy))

        return centroids