# quantize-rs

Neural network quantization toolkit for ONNX models, written in Rust with Python bindings.

[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Crates.io Downloads](https://img.shields.io/crates/d/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![PyPI](https://img.shields.io/pypi/v/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

quantize-rs converts float32 ONNX models to INT8 or INT4 representation using post-training quantization. It supports weight-only quantization, activation-based calibration, per-channel quantization, and outputs standard ONNX QDQ (DequantizeLinear) graphs compatible with ONNX Runtime.

> **Scope:** quantize-rs is designed and validated primarily for **computer-vision (CNN-style) ONNX models** -- ResNet, MobileNet, SqueezeNet, and similar architectures with Conv/MatMul backbones and image-shaped inputs. The weight-only path is model-agnostic and will quantize any FP32 ONNX initializer, but activation calibration uses [tract](https://github.com/sonos/tract) for inference, whose op coverage is geared toward CNNs. Transformer / LLM / sequence models may run into unsupported ops or shape-detection mismatches during calibration. See [Limitations](#known-limitations) for details.

## Features

- **INT8 and INT4 quantization** -- per-tensor or per-channel, asymmetric or symmetric
- **Native INT4 storage** -- `--native-int4` writes ONNX `DataType::Int4` (opset 21) for true 8x on-disk compression instead of widening to INT8 bytes
- **Symmetric quantization** -- `--symmetric` forces `zero_point == 0`, required by most ORT / TensorRT INT8 matmul kernels for per-channel weights
- **Activation-based calibration** -- runs real inference on calibration data via [tract](https://github.com/sonos/tract) to determine optimal quantization ranges
- **Multiple calibration methods** -- MinMax, Percentile (99.9th), Entropy (KL divergence), MSE -- with histogram-direct range optimization (no sample regeneration)
- **ONNX QDQ output format** -- quantized models use `DequantizeLinear` nodes and load directly in ONNX Runtime
- **Graph connectivity validation** -- verifies that every node input resolves after quantization
- **Per-layer selection** -- exclude layers by name, set per-layer bit widths, or skip small tensors via `min_elements`; available on `quantize`, `batch`, **and** `calibrate`
- **CLI** -- single-model quantization, batch processing, validation, benchmarking, config-file driven workflows; `validate` / `info` / `benchmark` support `--format json` for machine-readable output
- **Parallel batch processing** -- `--jobs N` quantizes multiple models concurrently
- **Optional memory-mapped loading** (`mmap` feature) -- streams multi-GB ONNX files without copying into RAM
- **Optional safetensors calibration input** (`safetensors-input` feature) -- load calibration data from HuggingFace `.safetensors` files
- **Python bindings** -- via PyO3; install with `pip install quantization-rs`
- **Typed error handling** -- `QuantizeError` enum at all public API boundaries (no more string-parsing `anyhow` errors)
- **Rust library** -- usable as a crate dependency; all public items have doc comments
- **Property-based tests** -- 17 proptest cases covering quantization round-trips, error bounds, and bit-packing
- **Criterion benchmarks** -- throughput and per-channel comparison benchmarks in `benches/`
- **Fuzz target** -- `cargo +nightly fuzz run onnx_load` stresses the protobuf decoder; see `fuzz/README.md`

## Installation

### Python

```bash
pip install quantization-rs
```

Build from source (requires Rust toolchain):

```bash
pip install maturin
git clone https://github.com/AR-Kamal/quantize-rs.git
cd quantize-rs
maturin develop --release --features python
```

### Rust CLI

```bash
cargo install quantize-rs
```

### As a library dependency

```toml
[dependencies]
quantize-rs = "0.8"
```

## Quick start

### Python

```python
import quantize_rs

# Weight-based INT8 quantization
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)

# INT4 with native opset-21 storage (true 8x compression)
quantize_rs.quantize("model.onnx", "model_int4.onnx", bits=4, native_int4=True)

# Symmetric per-channel INT8 (recommended for ORT/TensorRT matmul kernels)
quantize_rs.quantize("model.onnx", "out.onnx", bits=8, per_channel=True, symmetric=True)

# Activation-based calibration (better accuracy)
quantize_rs.quantize_with_calibration(
    "resnet18.onnx",
    "resnet18_int8.onnx",
    calibration_data="samples.npy",
    method="minmax"
)

# Inspect model metadata
info = quantize_rs.model_info("model.onnx")
print(f"{info.name}: {info.num_nodes} nodes")
```

See [Python API documentation](README_PYTHON.md) for the full reference.

### CLI

```bash
# INT8 quantization
quantize-rs quantize model.onnx -o model_int8.onnx

# INT4 with per-channel quantization
quantize-rs quantize model.onnx -o model_int4.onnx --bits 4 --per-channel

# INT4 with native opset-21 storage (true 8x on-disk compression)
quantize-rs quantize model.onnx -o model_int4.onnx --bits 4 --native-int4

# Symmetric quantization (zero_point == 0, ORT/TensorRT-friendly for matmul)
quantize-rs quantize model.onnx -o model_int8_sym.onnx --per-channel --symmetric

# Activation-based calibration
quantize-rs calibrate model.onnx \
    --data calibration.npy \
    -o model_calibrated.onnx \
    --method minmax

# Validate a quantized model (structure, connectivity, numerical sanity)
quantize-rs validate model.onnx model_int8.onnx --detailed

# Machine-readable output for tooling pipelines
quantize-rs validate model.onnx model_int8.onnx --format json
quantize-rs benchmark model.onnx model_int8.onnx --format json
quantize-rs info model.onnx --format json

# Compare original vs quantized
quantize-rs benchmark model.onnx model_int8.onnx

# Batch processing (4 models in parallel)
quantize-rs batch *.onnx -o quantized/ --bits 8 --per-channel --jobs 4

# Config-file driven workflow
quantize-rs config quantize.yaml --dry-run
```

### Rust library

```rust
use quantize_rs::{OnnxModel, Quantizer, QuantConfig};

fn main() -> anyhow::Result<()> {
    let mut model = OnnxModel::load("model.onnx")?;

    let config = QuantConfig {
        bits: 8,
        per_channel: true,
        symmetric: false,
        calibration_method: None,
        ..Default::default()
    };

    // `quantize_model` runs the full filter / parallel / layer_bits pipeline
    // and returns one `QuantizedWeightOutput` per kept weight.
    let outputs = Quantizer::new(config).quantize_model(&model)?;
    let quantized_data: Vec<_> = outputs.into_iter().map(|o| o.qdq).collect();

    model.save_quantized(&quantized_data, "model_int8.onnx")?;
    Ok(())
}
```

To opt into native INT4 storage (opset 21) on save:

```rust
use quantize_rs::onnx_utils::SaveOptions;

let opts = SaveOptions::default().with_native_int4(true);
model.save_quantized_with_options(&quantized_data, "model_int4.onnx", opts)?;
```

## CLI reference

### quantize

```
quantize-rs quantize <MODEL> [OPTIONS]

Options:
  -o, --output <FILE>             Output path [default: model_quantized.onnx]
  -b, --bits <4|8>                Bit width [default: 8]
      --per-channel               Per-channel quantization
      --symmetric                 Symmetric quantization (zero_point == 0)
      --native-int4               Store INT4 weights as ONNX DataType::Int4 (opset 21)
      --exclude <LAYER>           Exclude a layer by name (repeatable)
      --min-elements <N>          Skip tensors with fewer than N elements
      --layer-bits <LAYER=BITS>   Per-layer bit-width override (repeatable)
```

### calibrate

```
quantize-rs calibrate <MODEL> --data <DATA> [OPTIONS]

Options:
      --data <FILE>               Calibration data (.npy)
  -o, --output <FILE>             Output path [default: model_calibrated.onnx]
  -b, --bits <4|8>                Bit width [default: 8]
      --per-channel               Per-channel quantization
      --symmetric                 Symmetric quantization (zero_point == 0)
      --native-int4               Store INT4 weights as ONNX DataType::Int4 (opset 21)
      --method <METHOD>           minmax | percentile | entropy | mse [default: percentile]
      --exclude <LAYER>           Exclude a layer by name (repeatable)
      --min-elements <N>          Skip tensors with fewer than N elements
      --layer-bits <LAYER=BITS>   Per-layer bit-width override (repeatable)
```

### batch

```
quantize-rs batch <MODELS>... -o <DIR> [OPTIONS]

Options:
  -o, --output <DIR>              Output directory (required)
  -b, --bits <4|8>                Bit width [default: 8]
      --per-channel               Per-channel quantization
      --symmetric                 Symmetric quantization (zero_point == 0)
      --native-int4               Store INT4 weights as ONNX DataType::Int4 (opset 21)
      --jobs <N>                  Number of models to quantize in parallel [default: 1]
      --skip-existing             Skip models that already have output files
      --continue-on-error         Do not abort on individual model failures
      --exclude <LAYER>           Exclude a layer by name (repeatable)
      --min-elements <N>          Skip tensors with fewer than N elements
      --layer-bits <LAYER=BITS>   Per-layer bit-width override (repeatable)
```

### validate

```
quantize-rs validate <ORIGINAL> <QUANTIZED> [--detailed] [--format human|json]
```

Checks structure preservation, graph connectivity, weight shapes, and numerical sanity (all-zero detection, constant-value detection). With `--detailed`, prints per-layer error analysis. `--format json` emits a parseable report on stdout (banner suppressed).

### benchmark

```
quantize-rs benchmark <ORIGINAL> <QUANTIZED> [--format human|json]
```

Compares node counts, weight counts, file sizes, and compression ratios. The structure-preservation check is QDQ-aware: it accounts for the `DequantizeLinear` nodes inserted during the transform. `--format json` emits a parseable report on stdout.

### info

```
quantize-rs info <MODEL> [--format human|json]
```

Prints model name, opset version, node count, inputs, and outputs. `--format json` emits a parseable report on stdout.

### config

```
quantize-rs config <CONFIG_FILE> [--dry-run]
```

Runs quantization from a YAML or TOML configuration file. Example:

```yaml
bits: 8
per_channel: true

models:
  - input: models/resnet18.onnx
    output: quantized/resnet18_int8.onnx

  - input: models/mobilenet.onnx
    output: quantized/mobilenet_int8.onnx

batch:
  input_dir: "models/*.onnx"
  output_dir: quantized/
  skip_existing: true
```

## How it works

### Quantization

Each float32 weight tensor is mapped to a fixed-point integer representation:

```
scale = (max - min) / (qmax - qmin)
quantized = round(value / scale) + zero_point
dequantized = (quantized - zero_point) * scale
```

For INT8, the quantized range is [-128, 127]. For INT4, it is [-8, 7]. INT4 values are bit-packed (two values per byte) in memory for 8x compression. By default they are widened to INT8 bytes on disk (DequantizeLinear requires INT8 input in opsets < 21); pass `--native-int4` (CLI) or `native_int4=True` (Python) to write them as ONNX `DataType::Int4` in opset 21 for true 8x on-disk compression. The runtime must support opset 21 to load the native variant.

### Symmetric vs asymmetric

By default, quantization is asymmetric: `zero_point` is offset to fit the data range. Pass `--symmetric` to force `zero_point == 0` and use a balanced range `[-|max|, +|max|]`. Most ONNX Runtime / TensorRT INT8 matmul kernels require symmetric per-channel quantization for weights, so `--per-channel --symmetric` is the recommended combination when targeting accelerated INT8 inference.

### Per-channel quantization

Computes separate scale and zero_point for each output channel (axis 0). This is particularly effective when different channels have vastly different weight distributions, which is common in convolutional layers.

### Activation-based calibration

Instead of deriving quantization ranges from weight values alone, calibration runs forward passes on representative input samples and records the actual activation distributions at each layer. The observed ranges produce tighter quantization parameters. Four methods are available:

| Method | Strategy |
|--------|----------|
| MinMax | Use observed min/max directly |
| Percentile | Clip at 99.9th percentile to reduce outlier sensitivity |
| Entropy | Select the range that minimizes KL divergence between original and quantized distributions |
| MSE | Select the range that minimizes mean squared error |

### Output format

Quantized models use the ONNX QDQ pattern. For each quantized weight, the original float32 initializer is replaced with:

- `{name}_quantized` -- INT8 tensor (same shape)
- `{name}_scale` -- float32 scalar
- `{name}_zp` -- INT8 scalar
- A `DequantizeLinear` node whose output is the original tensor name

Because the DequantizeLinear output carries the original name, all downstream nodes (Conv, MatMul, etc.) remain unchanged. The graph loads and runs in ONNX Runtime without modification.

## ONNX Runtime integration

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_int8.onnx")
input_name = session.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: x})
```

## Testing

```bash
# Rust tests (136 passing on default features: 95 unit + 24 integration + 17 property-based)
cargo test

# All optional features (adds mmap + safetensors-input integration tests, ~139 total)
cargo test --all-features

# With output
cargo test -- --nocapture

# Integration tests requiring model files on disk
cargo test -- --ignored --nocapture

# Criterion benchmarks
cargo bench

# Fuzz the ONNX loader (nightly toolchain required)
cargo +nightly fuzz run onnx_load   # see fuzz/README.md

# Python tests (requires maturin develop)
pytest test_python_bindings.py -v
```

## Known limitations

- **ONNX input only.** PyTorch and TensorFlow models must be exported to ONNX first.
- **Vision models are the primary target.** Activation calibration is wired through `tract`, whose op coverage is centered on CNN architectures (Conv, MatMul, BatchNorm, ReLU, Pool, etc.). Transformer / LLM / RNN models with custom ops, dynamic shapes, KV-cache, or attention-mask plumbing may fail to load through tract or report unsupported ops during calibration. Weight-only quantization (`quantize` / `quantize_with_calibration` with no calibration data) does not use tract and works on any FP32 ONNX model.
- **Image-shaped default sample.** When calibration data is not provided, random samples default to `[3, 224, 224]` (CHW image). For other input layouts pass `--shape` (CLI examples) or `sample_shape=...` (Python).
- **Per-channel DequantizeLinear** writes 1-D scale/zero_point tensors with the `axis` attribute. ONNX Runtime supports this in opset >= 13.
- **Native INT4 storage** requires an ONNX runtime with opset 21 support. Without `--native-int4`, INT4 values are widened to INT8 bytes on disk (still 4x model-size reduction relative to FP32).
- **Single-input models** are assumed by the calibration path's auto shape detection. Multi-input graphs need an explicit `sample_shape` per input.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `cargo test` and `cargo clippy` pass
5. Submit a pull request

## License

[MIT](LICENSE)

## Acknowledgments

- [tract](https://github.com/sonos/tract) -- ONNX inference engine used for activation calibration
- [PyO3](https://github.com/PyO3/pyo3) -- Rust/Python interop
- [prost](https://github.com/tokio-rs/prost) + [protox](https://github.com/andrewhickman/protox) -- pure-Rust ONNX protobuf pipeline (no system `protoc` required)
