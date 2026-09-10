# quantize-rs Python API

Python bindings for the Rust ONNX quantization toolkit. Install with
`pip install quantization-rs`; build this checkout with
`maturin develop --features python`. Wheels use abi3-py39 (Python 3.9+).

This checkout is the unpublished **v0.10.0** candidate. Build this checkout to
test the candidate; registry installation commands install the published version.
See [release readiness](RELEASE_v0.10.0.md) and
[CALIBRATION.md](CALIBRATION.md) for migration from published v0.9 behavior.
The [labeled CNN evaluation](eval/CNN_EVALUATION.md) records MNIST accuracy,
latency and memory for the underlying Rust calibration pipeline.

The separate experimental [matrix activation prototype](MATRIX_CALIBRATION.md)
is currently a Rust/CLI API. Python activation calibration remains Conv-only.

## Weight-only quantization

The [bounded GPT-2 INT8 evaluation](eval/GPT2_EVALUATION.md) records matrix-weight
coverage, held-out perplexity and runtime behavior for the underlying Rust path.

```python
import quantize_rs
quantize_rs.quantize("model.onnx", "weights_int8.onnx", per_channel=True, symmetric=True)
quantize_rs.quantize("model.onnx", "weights_int4.onnx", bits=4, native_int4=True)
```

`quantize(input_path, output_path, bits=8, per_channel=False,
excluded_layers=None, min_elements=0, layer_bits=None, native_int4=False,
symmetric=False)`

| Parameter | Meaning |
|-----------|---------|
| `input_path`, `output_path` | ONNX source and destination paths |
| `bits` | 8 or 4 |
| `per_channel` | Conv axis 0; MatMul last weight dimension; Gemm axis 1 or 0 according to `transB` |
| `excluded_layers` | Initializer names to retain in FP32 |
| `min_elements` | Skip smaller tensors; default 0 |
| `layer_bits` | Initializer-name dictionary of 4/8 bit overrides |
| `native_int4` | Store INT4 natively at opset 21; otherwise widen to INT8 storage |
| `symmetric` | Force weight zero points to zero |

This path selects direct inline FP32 Conv/MatMul/Gemm weight initializers, outputs weight
DequantizeLinear nodes, and primarily reduces file size. Activation parameters do
not affect weight ranges. Runtime memory reduction or faster inference is not
guaranteed. External tensor files are unsupported.

Embeddings, biases and unrelated constants remain unchanged. Shared weights with
conflicting axes or unsupported uses return errors; exclusions keep them FP32.
Indirect weights through Reshape/Transpose are skipped. See
[operator selection and migration](OPERATOR_QUANTIZATION.md).

## Static INT8 Conv activation quantization

`quantize_with_calibration(input_path, output_path, calibration_data=None,
bits=8, per_channel=False, method="minmax", num_samples=100, sample_shape=None,
native_int4=False, symmetric=False, excluded_layers=None, min_elements=0,
layer_bits=None)`

```python
quantize_rs.quantize_with_calibration(
    "model.onnx", "calibrated.onnx", calibration_data="samples.npy",
    per_channel=True, symmetric=True, method="minmax",
    excluded_layers=["sensitive.weight"], min_elements=16,
    layer_bits={"conv.weight": 8},
)
```

This function runs real inference to measure selected Conv activation tensors,
then inserts activation QuantizeLinear/DequantizeLinear pairs. Conv weights use
their own ranges. Other operators remain floating point.

| Parameter | Meaning |
|-----------|---------|
| `calibration_data` | Required representative FP32 `.npy` data shaped `[samples,C,H,W]` |
| `bits` | Must be 8 |
| `per_channel`, `symmetric` | Weight settings; activations use scalar asymmetric INT8 parameters |
| `method` | `minmax`, `percentile`, `percentile:NN`, `entropy`, or `mse` |
| `excluded_layers` | Conv initializer names to exclude |
| `min_elements` | Skip smaller Conv weights |
| `layer_bits` | Conv initializer overrides; only 8 supported |
| `native_int4` | Must be False; use `quantize` for INT4 |
| `num_samples` | Legacy argument retained for compatibility; unused with data |
| `sample_shape` | Optional assertion that the supplied dataset has this sample shape |

Requires opset >= 13 and one fixed FP32 NCHW model input with batch 1. Missing
calibration data, dynamic/multiple inputs, existing QDQ, custom domains/subgraphs,
external tensors and unpreserved ONNX sections are rejected. Random fallback has
been removed. Excluded Conv nodes can still receive a quantized activation from a
selected upstream Conv. See [the full contract](CALIBRATION.md).

## Preparing data

Use representative preprocessed inputs from the deployment distribution. Each
sample excludes the batch dimension; save the stacked array as FP32:

```python
import numpy as np
samples = np.stack([preprocess(image) for image in calibration_images]).astype(np.float32)
np.save("samples.npy", samples)
```

Measure task accuracy on separate held-out data. There is no guaranteed accuracy
improvement over weight-only quantization or guaranteed inference speedup.

## Model information

`model_info(input_path)` returns `ModelInfo` with `name`, `version`,
`opset_version`, `num_nodes`, `inputs`, and `outputs`.

```python
info = quantize_rs.model_info("calibrated.onnx")
print(info.name, info.opset_version, info.inputs)
```

## Runtime integration and logging

```python
import onnxruntime as ort
session = ort.InferenceSession("calibrated.onnx", providers=["CPUExecutionProvider"])
output = session.run(None, {session.get_inputs()[0].name: your_fp32_batch})
```

Both quantization functions release the Python GIL during heavy work. Calls remain
synchronous; use an executor if invoking them from an asyncio event loop.

Warnings use Python logging via pyo3-log. Configure logger levels before the first
quantization call. The wheel includes `quantize_rs.pyi` and `py.typed` for editor
and type-checker support.

## Verification

After building and installing this checkout's wheel:

```bash
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs --python
```

On Windows append `.exe` to the binary path. The smoke test covers actual Python
keyword handling, filters, rejection behavior and ONNX Runtime output accuracy.

## License

[MIT](LICENSE)
