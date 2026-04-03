"""
Usage Examples for ASL Detector Optimization
Demonstrates different ways to use the optimized detector
"""

# Example 1: Basic Real-Time Detection
def example_basic_detection():
    """Simple real-time detection from webcam"""
    import cv2
    from asl_detector_optimized import AdaptiveASLDetector

    detector = AdaptiveASLDetector(frame_skip=2)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        cv2.imshow("ASL Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 2: With Hand Tracking
def example_with_hand_tracking():
    """Detection with MediaPipe hand tracking"""
    import cv2
    from asl_detector_optimized import AdaptiveASLDetector

    detector = AdaptiveASLDetector(
        use_hand_tracking=True,
        frame_skip=2
    )
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)

        if result["roi"]:
            roi, (x1, y1, x2, y2) = result["roi"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result["prediction"]:
            text = f"Prediction: {result['prediction']}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Motion: {result['motion']:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("ASL with Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 3: Fast Mode with Quantization
def example_fast_mode():
    """Maximum speed with quantized TFLite model"""
    import cv2
    from asl_detector_optimized import AdaptiveASLDetector

    detector = AdaptiveASLDetector(
        model_path="asl_model_quantized.tflite",
        use_tflite=True,
        frame_skip=3  # Skip more frames for speed
    )
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        cv2.imshow("Fast Mode", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 4: Video File Processing
def example_video_processing():
    """Process pre-recorded video file"""
    import cv2
    from asl_detector_optimized import AdaptiveASLDetector

    detector = AdaptiveASLDetector(frame_skip=2)
    cap = cv2.VideoCapture("video.mp4")

    frame_count = 0
    max_frames = 1000

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames: {detector.current_sentence}")

    cap.release()

    print(f"\nFinal Result: {detector.current_sentence}")
    print(f"Total frames: {frame_count}")
    print(f"Total predictions: {detector.stats['predictions_made']}")


# Example 5: Using Inference Engine Directly
def example_inference_engine():
    """Direct use of OptimizedInferenceEngine"""
    import cv2
    import numpy as np
    from inference_engine import OptimizedInferenceEngine

    engine = OptimizedInferenceEngine(
        model_path="asl_model_improved.h5",
        img_size=64,
        frame_skip=2,
        batch_size=1,
        use_tflite=False
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = cv2.resize(frame, (64, 64))
        roi = roi.astype(np.float32) / 255.0

        prediction = engine.predict(roi)

        if prediction is not None:
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]
            print(f"Class: {class_idx}, Confidence: {confidence:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 6: Async Inference with Threading
def example_async_inference():
    """Threaded inference for smooth UI"""
    import cv2
    from inference_engine import OptimizedInferenceEngine

    engine = OptimizedInferenceEngine(
        model_path="asl_model_improved.h5",
        batch_size=1
    )

    engine.start_inference_thread()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            engine.predict_async(frame)
            result = engine.get_result(timeout=0.01)

            if result is not None:
                print(f"Result: {result}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.stop_inference_thread()
        cap.release()
        cv2.destroyAllWindows()


# Example 7: Model Quantization
def example_quantization():
    """Quantize model to TFLite"""
    from model_quantization import quantize_model_to_tflite, convert_to_onnx, prune_model

    print("Converting models...")

    output_tflite = quantize_model_to_tflite("asl_model_improved.h5")
    print(f"✅ TFLite created: {output_tflite}")

    output_onnx = convert_to_onnx("asl_model_improved.h5")
    print(f"✅ ONNX created: {output_onnx}")

    output_pruned = prune_model("asl_model_improved.h5", pruning_rate=0.3)
    print(f"✅ Pruned model created: {output_pruned}")


# Example 8: Hand Tracking Only
def example_hand_tracking():
    """Use MediaPipe hand tracking independently"""
    import cv2
    from hand_tracking import MediaPipeHandTracker

    tracker = MediaPipeHandTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.detect_hands(frame)

        if detections["landmarks"]:
            print(f"Detected {len(detections['landmarks'])} hand(s)")

            for i, landmarks in enumerate(detections["landmarks"]):
                roi, coords = tracker.get_roi_from_hand(frame, landmarks)
                if roi is not None:
                    cv2.imshow(f"Hand {i}", roi)

            frame = tracker.draw_hand_landmarks(frame, detections)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


# Example 9: Motion Detection
def example_motion_detection():
    """Analyze hand motion"""
    import cv2
    from hand_tracking import MotionDetector

    motion = MotionDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion_score = motion.add_frame(frame)
        is_moving = motion.is_motion_detected()
        trend = motion.get_motion_trend()

        status = f"Motion: {motion_score:.2f} | Moving: {is_moving} | Trend: {trend}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 10: Performance Benchmarking
def example_benchmarking():
    """Measure performance of different configurations"""
    import cv2
    import time
    from asl_detector_optimized import AdaptiveASLDetector

    configurations = [
        {"name": "Original", "frame_skip": 1, "use_tflite": False},
        {"name": "Frame Skip 2", "frame_skip": 2, "use_tflite": False},
        {"name": "Frame Skip 3", "frame_skip": 3, "use_tflite": False},
        {"name": "TFLite", "frame_skip": 1, "use_tflite": True},
        {"name": "TFLite + Skip 2", "frame_skip": 2, "use_tflite": True},
    ]

    cap = cv2.VideoCapture(0)
    num_frames = 100

    results = {}

    for config in configurations:
        print(f"\nBenchmarking: {config['name']}...")

        try:
            detector = AdaptiveASLDetector(
                frame_skip=config["frame_skip"],
                use_tflite=config["use_tflite"]
            )

            start = time.time()

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                detector.process_frame(frame)

            elapsed = time.time() - start
            fps = num_frames / elapsed

            results[config["name"]] = {
                "fps": fps,
                "ms_per_frame": (elapsed / num_frames) * 1000,
                "time": elapsed,
            }

            print(f"  FPS: {fps:.1f}")
            print(f"  Ms/frame: {(elapsed/num_frames)*1000:.1f}ms")

        except Exception as e:
            print(f"  Error: {e}")

    cap.release()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"{name:20} | FPS: {metrics['fps']:6.1f} | Ms/frame: {metrics['ms_per_frame']:6.1f}")


# Example 11: Configuration Management
def example_configuration():
    """Use configuration file for settings"""
    from config import get_config, update_config

    print("Current Configuration:")
    print(f"Model Path: {get_config('MODEL_PATH')}")
    print(f"Frame Skip: {get_config('FRAME_SKIP')}")
    print(f"Confidence Threshold: {get_config('CONFIDENCE_THRESHOLD')}")

    print("\nUpdating configuration...")
    update_config('FRAME_SKIP', 3)
    update_config('CONFIDENCE_THRESHOLD', 0.7)

    print(f"Frame Skip: {get_config('FRAME_SKIP')}")
    print(f"Confidence Threshold: {get_config('CONFIDENCE_THRESHOLD')}")


if __name__ == "__main__":
    print("ASL Detector - Usage Examples")
    print("=" * 60)
    print("\nSelect an example to run:")
    print("1. Basic Detection")
    print("2. With Hand Tracking")
    print("3. Fast Mode")
    print("4. Video Processing")
    print("5. Inference Engine")
    print("6. Async Inference")
    print("7. Model Quantization")
    print("8. Hand Tracking Only")
    print("9. Motion Detection")
    print("10. Performance Benchmarking")
    print("11. Configuration Management")

    choice = input("\nEnter choice (1-11): ").strip()

    examples = {
        "1": example_basic_detection,
        "2": example_with_hand_tracking,
        "3": example_fast_mode,
        "4": example_video_processing,
        "5": example_inference_engine,
        "6": example_async_inference,
        "7": example_quantization,
        "8": example_hand_tracking,
        "9": example_motion_detection,
        "10": example_benchmarking,
        "11": example_configuration,
    }

    if choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("Invalid choice")
