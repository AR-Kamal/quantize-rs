# quantize-rs Python API

Quantize ONNX models from Python with the Rust quantize-rs engine. Supports INT8
and INT4 weight quantization for Conv/MatMul/Gemm, plus static INT8 activation
calibration for supported Conv models.

## Installation

Requires **Python 3.9+**:

```bash
python -m pip install quantization-rs
```

Install `quantization-rs`, then import `quantize_rs`. Prebuilt wheels use Python's
stable ABI (`abi3-py39`). ONNX Runtime and NumPy are separate dependencies for
inference and dataset preparation:

```bash
python -m pip install numpy onnxruntime
```

For source builds, use Rust 1.88+ and an active virtual environment. Follow the
[checkout and build instructions](https://github.com/AR-Kamal/quantize-rs/blob/master/README.md#build-this-checkout), including
`maturin develop --locked --release --features python`.

This reference describes the **0.10 API**. Calibration behavior changed from 0.9;
see [migration details](https://github.com/AR-Kamal/quantize-rs/blob/master/CALIBRATION.md) and the [changelog](https://github.com/AR-Kamal/quantize-rs/blob/master/CHANGELOG.md).
The experimental [matrix activation path](https://github.com/AR-Kamal/quantize-rs/blob/master/MATRIX_CALIBRATION.md) is a Rust/CLI API;
Python activation calibration remains Conv-only.

## Weight-only quantization

```python
import quantize_rs
quantize_rs.quantize("model.onnx", "weights_int8.onnx", per_channel=True, symmetric=True)
quantize_rs.quantize("model.onnx", "weights_int4.onnx", bits=4, native_int4=True)
```

**Signature:**

```python
quantize(input_path, output_path, bits=8, per_channel=False,
         excluded_layers=None, min_elements=0, layer_bits=None,
         native_int4=False, symmetric=False)
```

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
[operator selection and migration](https://github.com/AR-Kamal/quantize-rs/blob/master/OPERATOR_QUANTIZATION.md).

## Static INT8 Conv activation quantization

**Signature:**

```python
quantize_with_calibration(
    input_path, output_path, calibration_data=None, bits=8, per_channel=False,
    method="minmax", num_samples=100, sample_shape=None, native_int4=False,
    symmetric=False, excluded_layers=None, min_elements=0, layer_bits=None,
)
```

```python
import quantize_rs

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
selected upstream Conv. See [the full contract](https://github.com/AR-Kamal/quantize-rs/blob/master/CALIBRATION.md).

## Preparing data

Use representative preprocessed inputs from the deployment distribution. Define
`preprocess(image)` to match your model and return one `[C,H,W]` sample, without
the batch dimension. Given a collection of `calibration_images`, save the stacked
array as FP32:

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
import quantize_rs

info = quantize_rs.model_info("calibrated.onnx")
print(info.name, info.opset_version, info.inputs)
```

## Runtime integration and logging

After saving a preprocessed FP32 batch matching the model input as `input.npy`,
run inference:

```python
import numpy as np
import onnxruntime as ort

batch = np.load("input.npy")
options = ort.SessionOptions()
options.add_session_config_entry("session.x64quantprecision", "1")
session = ort.InferenceSession("calibrated.onnx", options, providers=["CPUExecutionProvider"])
output = session.run(None, {session.get_inputs()[0].name: batch})
```

For static INT8 models, the precision setting above avoids intermediate integer
saturation on x64 CPUs without VNNI while keeping graph optimization enabled.
The CPU calibration evaluations use this setting; see the
[CPU runtime guidance](https://github.com/AR-Kamal/quantize-rs/blob/master/CALIBRATION.md#cpu-runtime-precision).

Runtime fusion can change execution precision. In particular, ORT can fuse
weight DequantizeLinear/MatMul into a kernel that also quantizes activations.
See [runtime validation](https://github.com/AR-Kamal/quantize-rs/blob/master/OPERATOR_QUANTIZATION.md#validation) for FP32-activation
parity settings and separate default-runtime quality checks. Measure accuracy
and performance with the runtime settings you intend to deploy.

The [MNIST calibration report](https://github.com/AR-Kamal/quantize-rs/blob/master/eval/CNN_EVALUATION.md) and
[GPT-2 weight-only report](https://github.com/AR-Kamal/quantize-rs/blob/master/eval/GPT2_EVALUATION.md) include measured accuracy,
latency, and memory. Smaller files do not guarantee faster inference.

Both quantization functions release the Python GIL during heavy work. Calls remain
synchronous; use an executor if invoking them from an asyncio event loop.

Warnings use Python logging via pyo3-log. Configure logger levels before the first
quantization call. The wheel includes `quantize_rs.pyi` and `py.typed` for editor
and type-checker support.

## Verification

From the repository root, after installing this checkout's bindings and
`numpy`, `onnx`, and `onnxruntime`, build the CLI and run:

```bash
cargo build --locked --bin quantize-rs
python -m unittest discover -s eval -p test_operator_axes_smoke.py -v
python eval/operator_axes_smoke_test.py --binary target/debug/quantize-rs --python
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs --python
```

On Windows append `.exe` to the binary path. The suites cover operator
selection, axes, Python keyword handling, filters, rejection behavior, and ONNX
Runtime output accuracy.

## License

[MIT](https://github.com/AR-Kamal/quantize-rs/blob/master/LICENSE)
