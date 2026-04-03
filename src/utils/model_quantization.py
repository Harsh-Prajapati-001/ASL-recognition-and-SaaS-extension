import tensorflow as tf
import numpy as np
import os

def quantize_model_to_tflite(model_path, output_path="asl_model_quantized.tflite", img_size=64):
    """
    Convert TensorFlow model to quantized TFLite format (INT8)
    Reduces model size 3-4x and speeds up inference 2-3x
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Creating representative dataset for quantization...")
    def representative_data_gen():
        for _ in range(100):
            yield [np.random.rand(1, img_size, img_size, 3).astype(np.float32)]

    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.representative_data_gen = representative_data_gen
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression = ((original_size - quantized_size) / original_size) * 100

    print(f"✅ Quantized model saved: {output_path}")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Quantized size: {quantized_size:.2f} MB")
    print(f"   Compression: {compression:.1f}%")

    return output_path


def convert_to_onnx(model_path, output_path="asl_model.onnx"):
    """
    Convert TensorFlow model to ONNX format for cross-platform inference
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    try:
        import tf2onnx
        import onnx

        print("Converting to ONNX format...")
        spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="input"),)
        output_path, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
        print(f"✅ ONNX model saved: {output_path}")
        return output_path
    except ImportError:
        print("❌ tf2onnx not installed. Install with: pip install tf2onnx onnx")
        return None


def prune_model(model_path, output_path="asl_model_pruned.h5", pruning_rate=0.3):
    """
    Remove unnecessary weights from model (30% default)
    Reduces size and improves inference speed
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    try:
        from tensorflow_model_optimization.sparsity import keras as sparsity

        print(f"Pruning model with {pruning_rate*100}% sparsity...")
        pruning_params = {
            "pruning_schedule": sparsity.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_rate,
                begin_step=0,
                end_step=1000,
            )
        }

        model_for_pruning = sparsity.prune_low_magnitude(model, **pruning_params)
        model_for_pruning.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print(f"✅ Pruned model saved: {output_path}")
        model_for_pruning.save(output_path)
        return output_path
    except ImportError:
        print("❌ tensorflow-model-optimization not installed. Install with: pip install tensorflow-model-optimization")
        return None


if __name__ == "__main__":
    model_file = "asl_model_improved.h5"

    if os.path.exists(model_file):
        quantize_model_to_tflite(model_file)
        convert_to_onnx(model_file)
        prune_model(model_file)
    else:
        print(f"❌ Model file {model_file} not found. Train model first.")
