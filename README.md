# quantize-rs

Neural network quantization toolkit for ONNX models, written in Rust with Python bindings.

[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Crates.io Downloads](https://img.shields.io/crates/d/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![PyPI](https://img.shields.io/pypi/v/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

quantize-rs converts float32 ONNX models to INT8 or INT4 representation using post-training quantization. It supports weight-only quantization, activation-based calibration, per-channel quantization, and outputs standard ONNX QDQ (DequantizeLinear) graphs compatible with ONNX Runtime.

## Features

- **INT8 and INT4 quantization** -- per-tensor or per-channel
- **Activation-based calibration** -- runs real inference on calibration data via [tract](https://github.com/sonos/tract) to determine optimal quantization ranges
- **Multiple calibration methods** -- MinMax, Percentile (99.9th), Entropy (KL divergence), MSE
- **ONNX QDQ output format** -- quantized models use `DequantizeLinear` nodes and load directly in ONNX Runtime
- **Graph connectivity validation** -- verifies that every node input resolves after quantization
- **CLI** -- single-model quantization, batch processing, validation, benchmarking, config-file driven workflows
- **Python bindings** -- via PyO3; install with `pip install quantization-rs`
- **Typed error handling** -- `QuantizeError` enum at all public API boundaries (no more string-parsing `anyhow` errors)
- **Rust library** -- usable as a crate dependency; all public items have doc comments

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
quantize-rs = "0.5"
```

## Quick start

### Python

```python
import quantize_rs

# Weight-based INT8 quantization
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)

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

# Activation-based calibration
quantize-rs calibrate model.onnx \
    --data calibration.npy \
    -o model_calibrated.onnx \
    --method minmax

# Validate a quantized model (structure, connectivity, numerical sanity)
quantize-rs validate model.onnx model_int8.onnx --detailed

# Compare original vs quantized
quantize-rs benchmark model.onnx model_int8.onnx

# Batch processing
quantize-rs batch *.onnx -o quantized/ --bits 8 --per-channel

# Config-file driven workflow
quantize-rs config quantize.yaml --dry-run
```

### Rust library

```rust
use quantize_rs::{OnnxModel, Quantizer, QuantConfig};
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;

fn main() -> anyhow::Result<()> {
    let mut model = OnnxModel::load("model.onnx")?;
    let weights = model.extract_weights();

    let config = QuantConfig {
        bits: 8,
        per_channel: true,
        calibration_method: None,
    };
    let quantizer = Quantizer::new(config);

    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor(
            &weight.data,
            weight.shape.clone(),
        )?;
        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        let is_per_channel = quantized.is_per_channel();

        quantized_data.push(QdqWeightInput {
            original_name: weight.name.clone(),
            quantized_values: quantized.data(),
            scales,
            zero_points,
            bits: quantized.bits(),
            axis: if is_per_channel { Some(0) } else { None },
        });
    }

    model.save_quantized(&quantized_data, "model_int8.onnx")?;
    Ok(())
}
```

## CLI reference

### quantize

```
quantize-rs quantize <MODEL> [OPTIONS]

Options:
  -o, --output <FILE>     Output path [default: model_quantized.onnx]
  -b, --bits <4|8>        Bit width [default: 8]
      --per-channel       Per-channel quantization
```

### calibrate

```
quantize-rs calibrate <MODEL> --data <DATA> [OPTIONS]

Options:
      --data <FILE>       Calibration data (.npy)
  -o, --output <FILE>     Output path [default: model_calibrated.onnx]
  -b, --bits <4|8>        Bit width [default: 8]
      --per-channel       Per-channel quantization
      --method <METHOD>   minmax | percentile | entropy | mse [default: percentile]
```

### batch

```
quantize-rs batch <MODELS>... -o <DIR> [OPTIONS]

Options:
  -o, --output <DIR>      Output directory (required)
  -b, --bits <4|8>        Bit width [default: 8]
      --per-channel       Per-channel quantization
      --skip-existing     Skip models that already have output files
      --continue-on-error Do not abort on individual model failures
```

### validate

```
quantize-rs validate <ORIGINAL> <QUANTIZED> [--detailed]
```

Checks structure preservation, graph connectivity, weight shapes, and numerical sanity (all-zero detection, constant-value detection). With `--detailed`, prints per-layer error analysis.

### benchmark

```
quantize-rs benchmark <ORIGINAL> <QUANTIZED>
```

Compares node counts, weight counts, file sizes, and compression ratios.

### info

```
quantize-rs info <MODEL>
```

Prints model name, opset version, node count, inputs, and outputs.

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

For INT8, the quantized range is [-128, 127]. For INT4, it is [-8, 7]. INT4 values are bit-packed (two values per byte) in memory for 8x compression, but stored as INT8 in ONNX files (DequantizeLinear requires INT8 input in opsets < 21).

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
# Rust tests (57 unit + 6 integration tests)
cargo test

# With output
cargo test -- --nocapture

# Integration tests requiring model files on disk
cargo test -- --ignored --nocapture

# Python tests (requires maturin develop)
pytest test_python_bindings.py -v
```

## Known limitations

- ONNX input only. PyTorch and TensorFlow models must be exported to ONNX first.
- Per-channel DequantizeLinear writes 1-D scale/zero_point tensors with the `axis` attribute. ONNX Runtime supports this in opset >= 13.
- INT4 values are stored as INT8 bytes in the ONNX file. True 4-bit packing requires opset 21 or a custom operator.
- Quantizes all weight tensors. Per-layer selection is not yet supported.

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
- [protobuf](https://crates.io/crates/protobuf) / [onnx-rs](https://crates.io/crates/onnx) -- ONNX protobuf parsing
