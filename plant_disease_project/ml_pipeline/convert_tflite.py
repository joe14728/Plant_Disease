import tensorflow as tf
import os
import time
from dataset import load_dataset
from train import MODEL_DIR, MODEL_NAME

TFLITE_MODEL_NAME = "model_unoptimized.tflite"
TFLITE_QUANTIZED_MODEL_NAME = "model_quantized.tflite"

def convert_to_tflite():
    base_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(base_path):
        print(f"Error: Base model {base_path} not found.")
        return

    print(f"Loading Keras model from {base_path}...")
    model = tf.keras.models.load_model(base_path)

    print("\n--- 1. Converting to Unoptimized TFLite (Float32) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    unoptimized_path = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
    with open(unoptimized_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved unoptimized model to {unoptimized_path}")
    print(f"Size: {os.path.getsize(unoptimized_path) / (1024 * 1024):.2f} MB")

    print("\n--- 2. Converting to Quantized TFLite (INT8) ---")
    # For full integer quantization, we need a representative dataset
    _, _, test_ds, _, _ = load_dataset()
    
    def representative_dataset():
        for images, _ in test_ds.take(100):
            yield [tf.cast(images, tf.float32)]

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_dataset
    # Ensure all ops are quantized
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set input and output tensors to uint8 (optional, for fully quantized IO)
    converter_quant.inference_input_type = tf.uint8  
    converter_quant.inference_output_type = tf.uint8

    try:
        tflite_quant_model = converter_quant.convert()
        quantized_path = os.path.join(MODEL_DIR, TFLITE_QUANTIZED_MODEL_NAME)
        with open(quantized_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"Saved quantized model to {quantized_path}")
        print(f"Size: {os.path.getsize(quantized_path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error during INT8 quantization: {e}")
        print("Falling back to float16 quantization...")
        
        converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_fp16.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter_fp16.convert()
        
        fp16_path = os.path.join(MODEL_DIR, "model_fp16.tflite")
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16_model)
        print(f"Saved FP16 quantized model to {fp16_path}")
        print(f"Size: {os.path.getsize(fp16_path) / (1024 * 1024):.2f} MB")

def benchmark_inference():
    print("\n--- 3. Benchmarking Inference Speed ---")
    # Helper to run inference
    def run_inference(tflite_path, dataset, num_runs=50):
        if not os.path.exists(tflite_path):
             return None
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        is_quantized = input_details[0]['dtype'] == np.uint8
        
        times = []
        for images, _ in dataset.unbatch().take(num_runs):
            input_data = tf.expand_dims(images, 0)
            
            if is_quantized:
                 # Quantize input data to uint8
                 scale, zero_point = input_details[0]['quantization']
                 input_data = input_data / scale + zero_point
                 input_data = tf.cast(input_data, tf.uint8)
                 
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            start = time.time()
            interpreter.invoke()
            end = time.time()
            
            # interpreter.get_tensor(output_details[0]['index'])
            times.append((end - start) * 1000) # ms
            
        avg_time = sum(times) / len(times)
        return avg_time

    import numpy as np
    from dataset import load_dataset
    _, _, test_ds, _, _ = load_dataset()

    unop_path = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
    quant_path = os.path.join(MODEL_DIR, TFLITE_QUANTIZED_MODEL_NAME)
    fp16_path = os.path.join(MODEL_DIR, "model_fp16.tflite")

    print("Unoptimized Model (FP32):")
    avg_unop = run_inference(unop_path, test_ds)
    if avg_unop:
         print(f"  Avg inference time: {avg_unop:.2f} ms")
         
    print("\nQuantized Model (INT8):")
    avg_quant = run_inference(quant_path, test_ds)
    if avg_quant:
         print(f"  Avg inference time: {avg_quant:.2f} ms")
         
    print("\nQuantized Model (FP16):")
    avg_fp16 = run_inference(fp16_path, test_ds)
    if avg_fp16:
         print(f"  Avg inference time: {avg_fp16:.2f} ms")


if __name__ == "__main__":
    convert_to_tflite()
    benchmark_inference()
