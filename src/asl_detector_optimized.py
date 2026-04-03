import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
import os
from typing import Optional
from inference_engine import OptimizedInferenceEngine
from hand_tracking import MediaPipeHandTracker, MotionDetector


CATEGORIES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.6
ADAPTIVE_COOLDOWN_MIN = 0.5
ADAPTIVE_COOLDOWN_MAX = 2.0
STATIC_THRESHOLD = 0.01


class AdaptiveASLDetector:
    """
    Advanced ASL detector with:
    - MediaPipe hand tracking
    - Motion detection
    - Adaptive cooldown based on gesture confidence
    - Frame skipping and multi-threading
    """

    def __init__(
        self,
        model_path: str = "asl_model_improved.h5",
        use_tflite: bool = False,
        frame_skip: int = 2,
        use_hand_tracking: bool = True,
    ):
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.use_hand_tracking = use_hand_tracking

        self.engine = OptimizedInferenceEngine(
            model_path=model_path,
            img_size=IMG_SIZE,
            frame_skip=frame_skip,
            use_tflite=use_tflite,
        )

        self.hand_tracker = MediaPipeHandTracker() if use_hand_tracking else None
        self.motion_detector = MotionDetector()

        self.predictions_queue = deque(maxlen=15)
        self.current_sentence = ""
        self.last_pred = ""
        self.last_time = time.time()
        self.prev_landmarks = None

        self.stats = {
            "frames_processed": 0,
            "predictions_made": 0,
            "avg_confidence": 0.0,
        }

    def get_roi_from_frame(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Get ROI either from MediaPipe or fixed region

        Returns:
            (roi_image, (x1, y1, x2, y2)) or None
        """
        if self.use_hand_tracking and self.hand_tracker.available:
            detections = self.hand_tracker.detect_hands(frame)

            if detections["landmarks"]:
                landmarks = detections["landmarks"][0]
                roi, coords = self.hand_tracker.get_roi_from_hand(frame, landmarks, roi_size=IMG_SIZE, padding=0.15)

                if roi is not None:
                    return roi, coords, detections

        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        return roi, (x1, y1, x2, y2), {"landmarks": [], "confidence": []}

    def calculate_adaptive_cooldown(self, confidence: float) -> float:
        """
        Calculate cooldown based on prediction confidence
        High confidence = shorter cooldown
        Low confidence = longer cooldown
        """
        normalized_conf = max(0, min(1, confidence))
        cooldown = ADAPTIVE_COOLDOWN_MAX - (normalized_conf * (ADAPTIVE_COOLDOWN_MAX - ADAPTIVE_COOLDOWN_MIN))
        return cooldown

    def detect_gesture_start(self, motion_score: float, landmarks: list) -> bool:
        """
        Detect if user is starting a new gesture
        Returns True if hand is becoming static after motion
        """
        if not landmarks:
            return motion_score < 0.1

        is_static = self.motion_detector.is_motion_detected() == False
        return is_static

    def update_sentence(self, pred: str):
        """Update sentence with prediction"""
        if pred == "space":
            self.current_sentence += " "
        elif pred == "del":
            self.current_sentence = self.current_sentence[:-1]
        elif pred != "nothing":
            self.current_sentence += pred

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame

        Returns:
            dict with detection results
        """
        self.stats["frames_processed"] += 1
        motion_score = self.motion_detector.add_frame(frame)

        result = {
            "frame": frame,
            "prediction": None,
            "confidence": 0.0,
            "roi": None,
            "motion": motion_score,
            "is_static": False,
        }

        roi_data = self.get_roi_from_frame(frame)
        if roi_data is None:
            return result

        roi, coords, detections = roi_data
        result["roi"] = (roi, coords)

        prediction = self.engine.predict(roi)
        if prediction is None:
            return result

        class_index = np.argmax(prediction)
        confidence = float(prediction[class_index])
        pred_letter = CATEGORIES[class_index]

        result["prediction"] = pred_letter
        result["confidence"] = confidence
        result["is_static"] = len(detections["landmarks"]) > 0 and self.motion_detector.is_motion_detected() == False

        if confidence >= CONFIDENCE_THRESHOLD:
            self.predictions_queue.append((pred_letter, confidence))

            if len(self.predictions_queue) >= 5:
                most_common_pred = max(set([p[0] for p in self.predictions_queue]), key=lambda x: [p[0] for p in self.predictions_queue].count(x))
                avg_confidence = np.mean([p[1] for p in self.predictions_queue if p[0] == most_common_pred])

                current_time = time.time()
                adaptive_cooldown = self.calculate_adaptive_cooldown(avg_confidence)

                if most_common_pred != self.last_pred and (current_time - self.last_time) > adaptive_cooldown:
                    self.update_sentence(most_common_pred)
                    self.last_pred = most_common_pred
                    self.last_time = current_time
                    self.predictions_queue.clear()

                    self.stats["predictions_made"] += 1
                    self.stats["avg_confidence"] = avg_confidence

        return result

    def get_stats(self) -> dict:
        """Get detection statistics"""
        elapsed = time.time() - (time.time() - self.stats["frames_processed"] / 30)
        return {
            **self.stats,
            "sentence": self.current_sentence,
            "fps": self.engine.get_fps(time.time()),
        }


def run_webcam_detection(model_path: str = "asl_model_improved.h5", use_tflite: bool = False):
    """Run ASL detection on webcam"""
    print("Initializing detector...")
    detector = AdaptiveASLDetector(model_path=model_path, use_tflite=use_tflite, frame_skip=2)

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting detection (Press 'q' to quit, 'b' to backspace, 's' to save)")
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)

            frame_display = result["frame"]

            if result["roi"]:
                roi, (x1, y1, x2, y2) = result["roi"]
                frame_display = cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if result["prediction"]:
                text = f"Pred: {result['prediction']} ({result['confidence']:.2f})"
                cv2.putText(
                    frame_display,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            motion_color = (0, 255, 0) if result["is_static"] else (0, 165, 255)
            cv2.putText(
                frame_display,
                f"Motion: {result['motion']:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                motion_color,
                2,
            )

            cv2.putText(
                frame_display,
                f"Sentence: {detector.current_sentence}",
                (10, frame_display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            fps = detector.engine.frame_counter / (time.time() - start_time)
            cv2.putText(
                frame_display,
                f"FPS: {fps:.1f}",
                (frame_display.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            cv2.imshow("ASL Detection (Optimized)", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                detector.current_sentence = detector.current_sentence[:-1]
            elif key == ord("s"):
                with open("saved_sentences.txt", "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {detector.current_sentence}\n")
                detector.current_sentence = ""

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if detector.hand_tracker:
            detector.hand_tracker.release()

        print("\n✅ Detection stopped")
        print(f"Total predictions made: {detector.stats['predictions_made']}")
        print(f"Final sentence: {detector.current_sentence}")


if __name__ == "__main__":
    run_webcam_detection()
