import cv2
import numpy as np
import threading
import queue
from collections import deque
import time
from typing import Optional, Tuple
import os


class OptimizedInferenceEngine:
    """
    High-performance inference engine with:
    - Frame skipping (process every Nth frame)
    - Multi-threading (inference in separate thread)
    - Batch prediction support
    - GPU acceleration option
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 64,
        frame_skip: int = 3,
        batch_size: int = 1,
        use_tflite: bool = True,
        use_gpu: bool = False,
    ):
        """
        Args:
            model_path: Path to model (.tflite or .h5)
            img_size: Input image size
            frame_skip: Process every Nth frame (1=all, 3=every 3rd)
            batch_size: Frames to batch before prediction
            use_tflite: Use TFLite (faster) vs Keras
            use_gpu: Enable GPU acceleration
        """
        self.model_path = model_path
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.batch_size = batch_size
        self.use_tflite = use_tflite
        self.use_gpu = use_gpu

        self.frame_counter = 0
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.model = self._load_model()
        self.inference_thread = None
        self.running = False

    def _load_model(self):
        """Load model with appropriate backend"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self.use_tflite and self.model_path.endswith(".tflite"):
            return self._load_tflite_model()
        else:
            return self._load_keras_model()

    def _load_tflite_model(self):
        """Load TFLite model with optional GPU delegate"""
        import tensorflow as tf

        print(f"Loading TFLite model: {self.model_path}")

        if self.use_gpu:
            try:
                gpu_delegate = tf.lite.experimental.load_delegate("libgpu_delegate.so")
                interpreter = tf.lite.Interpreter(
                    model_path=self.model_path, experimental_delegates=[gpu_delegate]
                )
                print("✅ GPU delegate enabled")
            except Exception as e:
                print(f"⚠️  GPU delegate failed: {e}. Falling back to CPU.")
                interpreter = tf.lite.Interpreter(model_path=self.model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=self.model_path)

        interpreter.allocate_tensors()
        return interpreter

    def _load_keras_model(self):
        """Load Keras/H5 model"""
        import tensorflow as tf

        print(f"Loading Keras model: {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)

        if self.use_gpu:
            print("✅ GPU acceleration enabled for Keras model")

        return model

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame"""
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype(np.float32) / 255.0
        return frame

    def predict_batch(self, frames: list) -> np.ndarray:
        """Predict on batch of frames"""
        frames = np.array(frames)

        if isinstance(self.model, str) or hasattr(self.model, "get_signature"):
            return self._predict_tflite(frames)
        else:
            return self._predict_keras(frames)

    def _predict_tflite(self, frames: np.ndarray) -> np.ndarray:
        """TFLite prediction"""
        import tensorflow as tf

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        predictions = []
        for frame in frames:
            input_data = np.expand_dims(frame, axis=0)

            if input_details[0]["dtype"] == np.int8:
                input_scale, input_zero_point = (
                    input_details[0]["quantization"][0],
                    input_details[0]["quantization"][1],
                )
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

            self.model.set_tensor(input_details[0]["index"], input_data)
            self.model.invoke()

            output_data = self.model.get_tensor(output_details[0]["index"])

            if output_details[0]["dtype"] == np.int8:
                output_scale, output_zero_point = (
                    output_details[0]["quantization"][0],
                    output_details[0]["quantization"][1],
                )
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            predictions.append(output_data)

        return np.array(predictions)

    def _predict_keras(self, frames: np.ndarray) -> np.ndarray:
        """Keras prediction"""
        return self.model.predict(frames, verbose=0)

    def start_inference_thread(self):
        """Start background inference thread"""
        if self.running:
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("✅ Inference thread started")

    def stop_inference_thread(self):
        """Stop background inference thread"""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join()
        print("✅ Inference thread stopped")

    def _inference_loop(self):
        """Background inference loop"""
        batch = []

        while self.running:
            try:
                frame = self.inference_queue.get(timeout=0.1)
                batch.append(frame)

                if len(batch) >= self.batch_size:
                    predictions = self.predict_batch(batch)
                    self.result_queue.put(predictions)
                    batch = []

            except queue.Empty:
                if batch:
                    predictions = self.predict_batch(batch)
                    self.result_queue.put(predictions)
                    batch = []

    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Single frame prediction with frame skipping

        Returns:
            Prediction array or None if frame skipped
        """
        self.frame_counter += 1

        if self.frame_counter % self.frame_skip != 0:
            return None

        processed = self.preprocess(frame)
        prediction = self.predict_batch([processed])
        return prediction[0]

    def predict_async(self, frame: np.ndarray):
        """Add frame to async inference queue"""
        processed = self.preprocess(frame)
        self.inference_queue.put(processed)

    def get_result(self, timeout: float = 0.01) -> Optional[np.ndarray]:
        """Get result from async inference"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_fps(self, start_time: float) -> float:
        """Calculate FPS"""
        elapsed = time.time() - start_time
        return self.frame_counter / elapsed if elapsed > 0 else 0


class FrameBuffer:
    """Buffer frames for batch processing"""

    def __init__(self, max_size: int = 5):
        self.buffer = deque(maxlen=max_size)

    def add(self, frame: np.ndarray):
        """Add frame to buffer"""
        self.buffer.append(frame)

    def get_batch(self) -> Optional[list]:
        """Get all frames in buffer"""
        if len(self.buffer) > 0:
            return list(self.buffer)
        return None

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) == self.buffer.maxlen
