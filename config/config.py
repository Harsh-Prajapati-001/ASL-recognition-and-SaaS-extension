"""
Configuration file for ASL Detector
Centralized settings for easy customization
"""

# Model Configuration
MODEL_PATH = "asl_model_improved.h5"
USE_TFLITE = False  # Set to True after quantization
USE_GPU = False     # Set to True if GPU available

# Detection Configuration
IMG_SIZE = 64  # Change to 96 if using full model
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Performance Tuning
FRAME_SKIP = 2  # Process every Nth frame (1=all, 2=every 2nd, 3=every 3rd)
BATCH_SIZE = 1  # Increase for batch processing
USE_MULTITHREADING = False  # Enable for async inference

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.6  # 0.5=lenient, 0.6=balanced, 0.8=strict
ADAPTIVE_COOLDOWN_MIN = 0.5  # Minimum time between predictions (seconds)
ADAPTIVE_COOLDOWN_MAX = 2.0  # Maximum time between predictions (seconds)
STATIC_THRESHOLD = 0.01  # Hand velocity threshold for "static"

# Hand Tracking (MediaPipe)
USE_HAND_TRACKING = True  # Enable dynamic hand tracking
MAX_HANDS = 2  # Maximum number of hands to detect
MIN_DETECTION_CONFIDENCE = 0.5  # MediaPipe confidence threshold
HAND_ROI_PADDING = 0.15  # Padding around detected hand (0.15 = 15%)
HAND_ROI_SIZE = 64  # ROI extraction size

# Motion Detection
MOTION_HISTORY_SIZE = 5  # Frames to analyze for motion
MOTION_THRESHOLD = 20.0  # Threshold for motion detection

# Webcam Configuration
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30

# Video Processing
MAX_FRAMES_PER_VIDEO = 500
VIDEO_RESIZE_WIDTH = 640
VIDEO_RESIZE_HEIGHT = 480

# Inference Queue Settings
QUEUE_TIMEOUT = 0.01  # Timeout for queue operations (seconds)
PREDICTION_QUEUE_SIZE = 15  # Size of prediction smoothing queue

# Display Configuration
DISPLAY_FPS = True
DISPLAY_CONFIDENCE = True
DISPLAY_MOTION = True
DISPLAY_LANDMARKS = True
FONT = 'cv2.FONT_HERSHEY_SIMPLEX'
FONT_SCALE = 0.8
FONT_THICKNESS = 2
FONT_COLOR = (0, 255, 0)  # BGR: Green
BACKGROUND_COLOR = (255, 0, 0)  # BGR: Blue
TEXT_COLOR = (0, 255, 255)  # BGR: Yellow

# Output Configuration
SAVE_RESULTS = True
RESULTS_DIR = "results"
RESULTS_FILE = "saved_sentences.txt"
DETECTIONS_FILE = "detections.txt"

# Training Configuration
TRAIN_IMG_SIZE = 64
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 15
TRAIN_VALIDATION_SPLIT = 0.2
TRAIN_DATA_AUGMENTATION = True

# Data Augmentation Parameters
ROTATION_RANGE = 20
ZOOM_RANGE = 0.15
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.15
HORIZONTAL_FLIP = True

# Model Optimization
QUANTIZATION_ENABLED = False  # Set True to use quantized model
PRUNING_RATE = 0.3  # Remove 30% of weights
COMPRESSION_ENABLED = True

# Server Configuration
FLASK_HOST = "localhost"
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB

# Advanced Settings
PROFILE_PERFORMANCE = False  # Enable performance profiling
VERBOSE_LOGGING = True  # Print detailed logs
SAVE_FRAMES = False  # Save processed frames for analysis
DEMO_MODE = False  # Run without actual model


def get_config(key, default=None):
    """Get configuration value by key"""
    return globals().get(key, default)


def update_config(key, value):
    """Update configuration value"""
    if key in globals():
        globals()[key] = value
        return True
    return False


if __name__ == "__main__":
    print("ASL Detector Configuration")
    print("=" * 50)

    print("\nModel Settings:")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Use TFLite: {USE_TFLITE}")
    print(f"  Use GPU: {USE_GPU}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")

    print("\nPerformance Settings:")
    print(f"  Frame Skip: {FRAME_SKIP}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Multithreading: {USE_MULTITHREADING}")

    print("\nDetection Settings:")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Adaptive Cooldown: {ADAPTIVE_COOLDOWN_MIN}-{ADAPTIVE_COOLDOWN_MAX}s")
    print(f"  Hand Tracking: {USE_HAND_TRACKING}")

    print("\nVideo Settings:")
    print(f"  Webcam Size: {WEBCAM_WIDTH}x{WEBCAM_HEIGHT}")
    print(f"  Max Frames: {MAX_FRAMES_PER_VIDEO}")

    print("\nOutput Settings:")
    print(f"  Save Results: {SAVE_RESULTS}")
    print(f"  Results Dir: {RESULTS_DIR}")

    print("\nCategories: {0}".format(len(CATEGORIES)))
    print(f"  {CATEGORIES}")
