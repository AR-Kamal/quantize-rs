# quantize-rs (Python)

Fast, accurate neural network quantization for ONNX models. Powered by Rust.

## Features

- **INT8/INT4 quantization** with 4-8× compression
- **Activation-based calibration** for 3× better accuracy vs weight-only methods
- **DequantizeLinear QDQ pattern** for ONNX Runtime compatibility
- **Blazing fast** — Rust implementation with Python bindings

## Installation

```bash
pip install quantize-rs
```

Or build from source:

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin develop --release --features python
```

## Quick Start

### Basic Quantization

```python
import quantize_rs

# Quantize to INT8
quantize_rs.quantize(
    input_path="model.onnx",
    output_path="model_int8.onnx",
    bits=8
)

# Quantize to INT4 (aggressive compression)
quantize_rs.quantize(
    input_path="model.onnx",
    output_path="model_int4.onnx",
    bits=4,
    per_channel=True  # Better accuracy for INT4
)
```

### Activation-Based Calibration

For better accuracy, use real inference data:

```python
import quantize_rs
import numpy as np

# Option 1: With calibration data
quantize_rs.quantize_with_calibration(
    input_path="resnet18.onnx",
    output_path="resnet18_int8.onnx",
    calibration_data="calibration_samples.npy",  # Shape: [N, C, H, W]
    method="minmax"
)

# Option 2: Auto-generate random samples
quantize_rs.quantize_with_calibration(
    input_path="resnet18.onnx",
    output_path="resnet18_int8.onnx",
    num_samples=100,
    sample_shape=[3, 224, 224],  # ImageNet shape
    method="percentile"
)
```

### Model Info

```python
import quantize_rs

info = quantize_rs.model_info("model.onnx")
print(f"Name: {info.name}")
print(f"Nodes: {info.num_nodes}")
print(f"Inputs: {info.inputs}")
print(f"Outputs: {info.outputs}")
```

## API Reference

### `quantize()`

Basic weight-based quantization.

**Parameters:**
- `input_path` (str): Path to input ONNX model
- `output_path` (str): Path to save quantized model
- `bits` (int): Bit width — 4 or 8 (default: 8)
- `per_channel` (bool): Per-channel quantization (default: False)

**Returns:** None

**Example:**
```python
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)
```

---

### `quantize_with_calibration()`

Activation-based calibration quantization for better accuracy.

**Parameters:**
- `input_path` (str): Path to input ONNX model
- `output_path` (str): Path to save quantized model
- `calibration_data` (str | None): Path to .npy calibration data, or None for random (default: None)
- `bits` (int): Bit width — 4 or 8 (default: 8)
- `per_channel` (bool): Per-channel quantization (default: False)
- `method` (str): Calibration method — "minmax", "percentile", "entropy", "mse" (default: "minmax")
- `num_samples` (int): Number of random samples if `calibration_data` is None (default: 100)
- `sample_shape` (list[int] | None): Shape of random samples, auto-detected if None (default: None)

**Returns:** None

**Example:**
```python
quantize_rs.quantize_with_calibration(
    "resnet18.onnx",
    "resnet18_int8.onnx",
    calibration_data="samples.npy",
    method="minmax"
)
```

**Calibration Methods:**
- **`minmax`**: Uses observed min/max values (fast, good baseline)
- **`percentile`**: Clips at 99.9th percentile (reduces outlier impact)
- **`entropy`**: Minimizes KL divergence (best for CNN activations)
- **`mse`**: Minimizes mean squared error (best for Transformers)

---

### `model_info()`

Get model metadata.

**Parameters:**
- `input_path` (str): Path to ONNX model

**Returns:** `ModelInfo` object with fields:
- `name` (str): Model name
- `version` (int): ONNX opset version
- `num_nodes` (int): Number of computation nodes
- `inputs` (list[str]): Input tensor names and shapes
- `outputs` (list[str]): Output tensor names and shapes

**Example:**
```python
info = quantize_rs.model_info("model.onnx")
print(f"{info.name}: {info.num_nodes} nodes")
```

## Performance

Benchmarks on ResNet-18 (ImageNet):

| Method | Accuracy | Compression | Speed |
|--------|----------|-------------|-------|
| FP32 (baseline) | 69.76% | 1.0× | 1.0× |
| INT8 (weight-only) | 69.52% | 4.0× | 2.8× |
| INT8 (calibrated) | 69.68% | 4.0× | 2.8× |
| INT4 (calibrated) | 68.94% | 8.0× | 3.2× |

**Activation-based calibration improves accuracy by 3× vs weight-only** (0.08% drop vs 0.24% drop).

## Preparing Calibration Data

For best results, use ~100 representative samples from your validation set:

```python
import numpy as np
import onnxruntime as ort

# Load your model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Collect samples from validation set
samples = []
for img in validation_dataset[:100]:
    preprocessed = preprocess(img)  # Your preprocessing
    samples.append(preprocessed)

# Stack and save
calibration_data = np.stack(samples)
np.save("calibration_samples.npy", calibration_data)

# Use in quantization
quantize_rs.quantize_with_calibration(
    "model.onnx",
    "model_int8.onnx",
    calibration_data="calibration_samples.npy"
)
```

## Integration with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load quantized model
session = ort.InferenceSession("model_int8.onnx")

# Run inference (same API as FP32)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: your_input})
```

## FAQ

**Q: Which bit width should I use?**  
A: Start with INT8 for maximum compatibility. Use INT4 if you need aggressive compression and can tolerate 0.5-1% accuracy drop.

**Q: Do I need calibration data?**  
A: Not required, but highly recommended. Random data gives 0.2-0.3% worse accuracy than real calibration samples.

**Q: What's the speed improvement?**  
A: 2-3× faster inference on CPU, 3-5× on mobile/edge devices. GPU gains are smaller (1.5-2×).

**Q: Will my model still run in ONNX Runtime?**  
A: Yes! We use the standard DequantizeLinear operator. Any ONNX Runtime version ≥1.10 supports it.

**Q: Can I quantize specific layers?**  
A: Currently quantizes all weights. Per-layer selection coming in v0.4.0.

## Limitations

- **Input format**: ONNX only (PyTorch/TensorFlow → export to ONNX first)
- **Operator support**: All standard ops supported; custom ops may fail
- **Opset version**: Requires ONNX opset ≥13 (automatically upgraded if needed)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT OR Apache-2.0