import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import queue
from io import BytesIO
from datetime import datetime
import json

from asl_detector_optimized import AdaptiveASLDetector

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

detector = None
processing = False
detection_results = {"text": "", "confidence": [], "frames": 0}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available models"""
    models = []
    for file in os.listdir("."):
        if file.endswith(".h5") or file.endswith(".tflite"):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            models.append({"name": file, "size": f"{size_mb:.2f} MB"})

    return jsonify(models)


@app.route("/api/init", methods=["POST"])
def init_detector():
    """Initialize detector with selected model"""
    global detector

    data = request.json
    model_path = data.get("model", "asl_model_improved.h5")
    use_tflite = data.get("use_tflite", False)
    frame_skip = data.get("frame_skip", 2)

    try:
        detector = AdaptiveASLDetector(
            model_path=model_path,
            use_tflite=use_tflite,
            frame_skip=frame_skip,
        )
        return jsonify({"status": "success", "message": f"Detector initialized with {model_path}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/process-video", methods=["POST"])
def process_video():
    """Process uploaded video or webcam stream"""
    global processing, detection_results, detector

    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialized"}), 400

    if processing:
        return jsonify({"status": "error", "message": "Already processing"}), 400

    source = request.form.get("source", "upload")
    max_frames = int(request.form.get("max_frames", 500))

    processing = True
    detection_results = {"text": "", "confidence": [], "frames": 0}

    try:
        if source == "upload":
            if "file" not in request.files:
                return jsonify({"status": "error", "message": "No file uploaded"}), 400

            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            thread = threading.Thread(target=_process_video_file, args=(filepath, max_frames))
            thread.start()

            return jsonify({"status": "processing", "message": "Video processing started"})

        elif source == "webcam":
            thread = threading.Thread(target=_process_webcam, args=(max_frames,))
            thread.start()

            return jsonify({"status": "processing", "message": "Webcam capture started"})

    except Exception as e:
        processing = False
        return jsonify({"status": "error", "message": str(e)}), 400


def _process_video_file(filepath, max_frames):
    """Process video file in background"""
    global detector, processing, detection_results

    try:
        cap = cv2.VideoCapture(filepath)

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            result = detector.process_frame(frame)

            if result["prediction"]:
                detection_results["confidence"].append(result["confidence"])

            detection_results["frames"] = frame_count
            detection_results["text"] = detector.current_sentence

            frame_count += 1

        cap.release()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("results/detections.txt", "a") as f:
            f.write(f"[{timestamp}] {detection_results['text']}\n")

        processing = False

    except Exception as e:
        print(f"Error processing video: {e}")
        processing = False


def _process_webcam(duration_frames):
    """Process webcam stream in background"""
    global detector, processing, detection_results

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        while cap.isOpened() and frame_count < duration_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)

            if result["prediction"]:
                detection_results["confidence"].append(result["confidence"])

            detection_results["frames"] = frame_count
            detection_results["text"] = detector.current_sentence

            frame_count += 1

        cap.release()
        processing = False

    except Exception as e:
        print(f"Error processing webcam: {e}")
        processing = False


@app.route("/api/progress", methods=["GET"])
def get_progress():
    """Get processing progress"""
    global processing, detection_results

    avg_confidence = (
        np.mean(detection_results["confidence"]) if detection_results["confidence"] else 0.0
    )

    return jsonify({
        "processing": processing,
        "frames": detection_results["frames"],
        "text": detection_results["text"],
        "confidence": avg_confidence,
    })


@app.route("/api/save", methods=["POST"])
def save_result():
    """Save detection result"""
    global detection_results

    data = request.json
    custom_text = data.get("text", detection_results["text"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/saved_detections.txt", "a") as f:
        f.write(f"[{timestamp}] {custom_text}\n")

    return jsonify({"status": "success", "message": "Result saved"})


@app.route("/api/clear", methods=["POST"])
def clear_result():
    """Clear current result"""
    global detector, detection_results

    if detector:
        detector.current_sentence = ""

    detection_results = {"text": "", "confidence": [], "frames": 0}

    return jsonify({"status": "success"})


@app.route("/api/quantize", methods=["POST"])
def quantize_model():
    """Quantize model to TFLite"""
    data = request.json
    model_path = data.get("model", "asl_model_improved.h5")

    try:
        from model_quantization import quantize_model_to_tflite

        output_path = quantize_model_to_tflite(model_path)
        return jsonify({
            "status": "success",
            "message": f"Model quantized successfully",
            "output": output_path,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get detection history"""
    history = []

    if os.path.exists("results/saved_detections.txt"):
        with open("results/saved_detections.txt", "r") as f:
            history = f.readlines()[-20:]

    return jsonify({"history": history})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
