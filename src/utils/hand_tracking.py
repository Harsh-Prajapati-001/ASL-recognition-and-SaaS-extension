import cv2
import numpy as np
from typing import Tuple, Optional, List
import math


class MediaPipeHandTracker:
    """
    Hand detection and tracking using MediaPipe
    Replaces fixed ROI with dynamic hand tracking
    """

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        """
        Args:
            static_image_mode: If True, detection runs on each frame independently
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence threshold
        """
        try:
            import mediapipe as mp

            self.mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.available = True
        except ImportError:
            print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")
            self.available = False

    def detect_hands(self, frame: np.ndarray) -> dict:
        """
        Detect hands in frame

        Returns:
            dict with keys:
                - 'hands': list of hand detections
                - 'landmarks': list of 21 landmarks per hand
                - 'handedness': 'Right' or 'Left'
                - 'confidence': detection confidence (0-1)
        """
        if not self.available:
            return {"hands": [], "landmarks": [], "handedness": [], "confidence": []}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        detections = {"hands": [], "landmarks": [], "handedness": [], "confidence": []}

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                handedness = hand_info.classification[0].label
                confidence = hand_info.classification[0].score

                detections["landmarks"].append(landmarks)
                detections["handedness"].append(handedness)
                detections["confidence"].append(confidence)
                detections["hands"].append(hand_landmarks)

        return detections

    def get_roi_from_hand(self, frame: np.ndarray, landmarks: List[Tuple], roi_size: int = 128, padding: float = 0.2) -> Tuple[np.ndarray, Tuple]:
        """
        Extract ROI around detected hand

        Args:
            frame: Input frame
            landmarks: List of (x, y, z) hand landmarks
            roi_size: Output ROI size
            padding: Padding around hand (0.2 = 20% extra space)

        Returns:
            (roi_image, (x1, y1, x2, y2)) - extracted ROI and its coordinates
        """
        if not landmarks:
            return None, (0, 0, 0, 0)

        h, w = frame.shape[:2]

        xs = [lm[0] * w for lm in landmarks]
        ys = [lm[1] * h for lm in landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        hand_width = x_max - x_min
        hand_height = y_max - y_min

        x_min -= hand_width * padding
        x_max += hand_width * padding
        y_min -= hand_height * padding
        y_max += hand_height * padding

        x1 = max(0, int(x_min))
        y1 = max(0, int(y_min))
        x2 = min(w, int(x_max))
        y2 = min(h, int(y_max))

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None, (0, 0, 0, 0)

        roi = cv2.resize(roi, (roi_size, roi_size))
        return roi, (x1, y1, x2, y2)

    def draw_hand_landmarks(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw hand landmarks on frame"""
        if not self.available or not detections["hands"]:
            return frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for hand_landmarks in detections["hands"]:
            self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp.solutions.hands.HAND_CONNECTIONS)

        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def draw_roi_box(self, frame: np.ndarray, roi_coords: Tuple, color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw ROI bounding box on frame"""
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    def get_hand_velocity(self, prev_landmarks: List[Tuple], curr_landmarks: List[Tuple]) -> float:
        """
        Calculate hand movement velocity

        Returns:
            Average distance moved by hand landmarks
        """
        if not prev_landmarks or not curr_landmarks:
            return 0.0

        distances = []
        for prev, curr in zip(prev_landmarks, curr_landmarks):
            dist = math.sqrt((prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2)
            distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def is_hand_static(self, velocity: float, threshold: float = 0.01) -> bool:
        """Check if hand is static (low velocity)"""
        return velocity < threshold

    def release(self):
        """Release MediaPipe resources"""
        if self.available:
            self.hands.close()


class MotionDetector:
    """Detect motion in frames for gesture recognition"""

    def __init__(self, history_size: int = 5, threshold: float = 20.0):
        """
        Args:
            history_size: Number of frames to store for motion calculation
            threshold: Motion threshold (0-255 pixel difference)
        """
        self.history = []
        self.history_size = history_size
        self.threshold = threshold
        self.prev_frame = None

    def add_frame(self, frame: np.ndarray) -> float:
        """
        Add frame and return motion magnitude

        Returns:
            Motion score (0 = no motion, 1 = high motion)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0

        frame_diff = cv2.absdiff(self.prev_frame, gray)
        motion = np.mean(frame_diff)

        self.prev_frame = gray
        self.history.append(motion)

        if len(self.history) > self.history_size:
            self.history.pop(0)

        return min(motion / 255.0, 1.0)

    def is_motion_detected(self) -> bool:
        """Check if significant motion detected"""
        if not self.history:
            return False
        return np.mean(self.history) > self.threshold

    def get_motion_trend(self) -> str:
        """
        Get motion trend

        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        if len(self.history) < 2:
            return "stable"

        recent = np.mean(self.history[-2:])
        previous = np.mean(self.history[:2])

        if recent > previous * 1.2:
            return "increasing"
        elif recent < previous * 0.8:
            return "decreasing"
        return "stable"

    def reset(self):
        """Reset motion history"""
        self.history = []
        self.prev_frame = None
