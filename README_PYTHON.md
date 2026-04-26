# quantize-rs Python API

Python bindings for quantize-rs, a neural network quantization toolkit for ONNX models.

## Scope

quantize-rs is designed and validated primarily for **computer-vision (CNN-style) ONNX models** -- ResNet, MobileNet, SqueezeNet, and similar architectures. Weight-only quantization (`quantize()`) is model-agnostic and works on any FP32 ONNX file. Activation calibration (`quantize_with_calibration()`) runs inference through [tract](https://github.com/sonos/tract), whose op coverage is centered on CNNs; transformer / LLM / RNN models may fail to load through tract or hit unsupported ops during calibration.

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

### `quantize(input_path, output_path, bits=8, per_channel=False, excluded_layers=None, min_elements=0, layer_bits=None, native_int4=False, symmetric=False)`

Weight-based quantization. Loads the model, quantizes all weight tensors, and saves the result in ONNX QDQ format.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_path` | str | required | Path to input ONNX model |
| `output_path` | str | required | Path to save quantized model |
| `bits` | int | 8 | Bit width: 4 or 8 |
| `per_channel` | bool | False | Use per-channel quantization (separate scale/zp per output channel) |
| `excluded_layers` | list[str] or None | None | Initializer names to leave in FP32 |
| `min_elements` | int | 0 | Skip tensors with fewer than N elements (e.g., biases) |
| `layer_bits` | dict[str, int] or None | None | Per-layer bit-width overrides, e.g. `{"conv1.weight": 4}` |
| `native_int4` | bool | False | Store INT4 weights as ONNX `DataType.Int4` (opset 21). True 8x on-disk compression but requires opset-21 runtime. No effect on INT8-only models. |
| `symmetric` | bool | False | Symmetric quantization (`zero_point == 0`). Required by most ORT / TensorRT INT8 matmul kernels for per-channel weights. |

**Example:**

```python
import quantize_rs

# Plain INT8
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)

# INT4 with native opset-21 storage (8x on-disk)
quantize_rs.quantize("model.onnx", "model_int4.onnx", bits=4, native_int4=True)

# Symmetric per-channel INT8 for ORT INT8 matmul kernels
quantize_rs.quantize(
    "model.onnx",
    "model_int8_sym.onnx",
    bits=8,
    per_channel=True,
    symmetric=True,
)

# Mixed precision: some layers INT4, rest INT8
quantize_rs.quantize(
    "model.onnx",
    "out.onnx",
    bits=8,
    layer_bits={"fc.weight": 4},
    excluded_layers=["embedding.weight"],
    min_elements=1024,  # skip small tensors (biases) and keep them FP32
)
```

---

### `quantize_with_calibration(input_path, output_path, calibration_data=None, bits=8, per_channel=False, method="minmax", num_samples=100, sample_shape=None, native_int4=False, symmetric=False)`

Activation-based calibration quantization. Runs inference on calibration samples to determine optimal quantization ranges per layer, then quantizes using those ranges. The full filter pipeline (`excluded_layers`, `min_elements`, `layer_bits`) is honored; pass these via `quantize()` directly if you need to skip layers explicitly.

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
| `sample_shape` | list[int] or None | None | Shape of random samples; auto-detected from model if None. Default fallback is `[3, 224, 224]` (CHW image) -- override for non-image inputs. |
| `native_int4` | bool | False | Store INT4 weights as ONNX `DataType.Int4` (opset 21) |
| `symmetric` | bool | False | Symmetric quantization (`zero_point == 0`) |

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
- Validated primarily on CNN-style vision models. Activation calibration uses tract for inference; transformer / LLM / RNN architectures may report unsupported ops or shape mismatches in `quantize_with_calibration()`. The plain `quantize()` (weight-only) function does not use tract and works on any FP32 ONNX model.
- Requires ONNX opset >= 10 for per-tensor quantization, >= 13 for per-channel (automatically upgraded if needed).
- INT4 values are stored as INT8 bytes by default. Pass `native_int4=True` to write them as ONNX `DataType.Int4` (opset 21) for true 8x compression -- requires an ONNX runtime with opset-21 support.
- Single-input models are assumed by random-sample auto shape detection; for multi-input graphs, pass `sample_shape` explicitly or supply real `calibration_data`.

## License

[MIT](LICENSE)
