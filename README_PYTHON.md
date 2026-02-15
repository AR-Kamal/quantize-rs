# quantize-rs Python API

Python bindings for quantize-rs, a neural network quantization toolkit for ONNX models.

## Installation

```bash
pip install quantization-rs
```

Build from source (requires Rust toolchain and maturin):

```bash
pip install maturin
maturin develop --release --features python
```

## API reference

### `quantize(input_path, output_path, bits=8, per_channel=False)`

Weight-based quantization. Loads the model, quantizes all weight tensors, and saves the result in ONNX QDQ format.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_path` | str | required | Path to input ONNX model |
| `output_path` | str | required | Path to save quantized model |
| `bits` | int | 8 | Bit width: 4 or 8 |
| `per_channel` | bool | False | Use per-channel quantization (separate scale/zp per output channel) |

**Example:**

```python
import quantize_rs

quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)
quantize_rs.quantize("model.onnx", "model_int4.onnx", bits=4, per_channel=True)
```

---

### `quantize_with_calibration(input_path, output_path, ...)`

Activation-based calibration quantization. Runs inference on calibration samples to determine optimal quantization ranges per layer, then quantizes using those ranges.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_path` | str | required | Path to input ONNX model |
| `output_path` | str | required | Path to save quantized model |
| `calibration_data` | str or None | None | Path to `.npy` file (shape `[N, ...]`), or None for random samples |
| `bits` | int | 8 | Bit width: 4 or 8 |
| `per_channel` | bool | False | Per-channel quantization |
| `method` | str | "minmax" | Calibration method (see below) |
| `num_samples` | int | 100 | Number of random samples when `calibration_data` is None |
| `sample_shape` | list[int] or None | None | Shape of random samples; auto-detected from model if None |

**Calibration methods:**

| Method | Description |
|--------|-------------|
| `"minmax"` | Uses observed min/max from activations |
| `"percentile"` | Clips at 99.9th percentile to reduce outlier sensitivity |
| `"entropy"` | Selects range minimizing KL divergence between original and quantized distributions |
| `"mse"` | Selects range minimizing mean squared error |

**Example:**

```python
import quantize_rs

# With real calibration data
quantize_rs.quantize_with_calibration(
    "resnet18.onnx",
    "resnet18_int8.onnx",
    calibration_data="calibration_samples.npy",
    method="minmax"
)

# With random samples (auto-detects input shape from model)
quantize_rs.quantize_with_calibration(
    "resnet18.onnx",
    "resnet18_int8.onnx",
    num_samples=100,
    sample_shape=[3, 224, 224],
    method="percentile"
)
```

---

### `model_info(input_path)`

Returns metadata about an ONNX model.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_path` | str | required | Path to ONNX model |

**Returns:** `ModelInfo` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Graph name |
| `version` | int | Model version |
| `num_nodes` | int | Number of computation nodes |
| `inputs` | list[str] | Input tensor names |
| `outputs` | list[str] | Output tensor names |

**Example:**

```python
info = quantize_rs.model_info("model.onnx")
print(f"Name: {info.name}")
print(f"Nodes: {info.num_nodes}")
print(f"Inputs: {info.inputs}")
print(f"Outputs: {info.outputs}")
```

## Preparing calibration data

For best results, use 50-200 representative samples from your validation or training set:

```python
import numpy as np

# Collect preprocessed samples
samples = []
for img in validation_dataset[:100]:
    preprocessed = preprocess(img)  # your preprocessing pipeline
    samples.append(preprocessed)

# Save as .npy (shape: [num_samples, channels, height, width])
calibration_data = np.stack(samples)
np.save("calibration_samples.npy", calibration_data)

# Use during quantization
quantize_rs.quantize_with_calibration(
    "model.onnx",
    "model_int8.onnx",
    calibration_data="calibration_samples.npy",
    method="minmax"
)
```

If you do not have calibration data, the function generates random samples. This is adequate for testing but will produce less accurate quantization than real data.

## ONNX Runtime integration

Quantized models use the standard `DequantizeLinear` operator and load directly in ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_int8.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: your_input})
```

## Limitations

- ONNX format only. Export PyTorch/TensorFlow models to ONNX before quantizing.
- Requires ONNX opset >= 13 (automatically upgraded if needed).
- INT4 values are stored as INT8 bytes in the ONNX file (DequantizeLinear requires INT8 input in opsets < 21).
- All weight tensors are quantized. Per-layer selection is not yet supported.

## License

[MIT](LICENSE)
