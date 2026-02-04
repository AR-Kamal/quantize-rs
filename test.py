import onnxruntime as ort
import numpy as np

# Try to load the quantized model
print("Loading quantized model...")
session = ort.InferenceSession('model_int8_calibrated.onnx')
print("✓ Model loaded successfully!")

# Get input info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input: {input_name}, shape: {input_shape}")

# Try inference
print("\nRunning inference...")
x = np.random.randn(1, 1, 28, 28).astype(np.float32)
output = session.run(None, {input_name: x})
print(f"✓ Inference works! Output shape: {output[0].shape}")
print(f"Output values (first 5): {output[0][0][:5]}")