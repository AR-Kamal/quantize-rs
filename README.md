# quantize-rs

> Fast neural network quantization for ONNX models — now with Python support

[![PyPI](https://img.shields.io/pypi/v/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Crates.io Downloads](https://img.shields.io/crates/d/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**quantize-rs** compresses neural networks by **4-8×** while maintaining accuracy. Convert float32 ONNX models to INT8/INT4 with activation-based calibration for optimal quality.

---

## What's New in v0.3.0

- **Python bindings** - Use from Python with `pip install quantization-rs`
- **Activation-based calibration** - Real inference for 3× better accuracy vs weight-only
- **ONNX Runtime compatibility** - Quantized models load and run in ONNX Runtime
- **DequantizeLinear pattern** - Standard ONNX QDQ format for broad compatibility

---

## Features

- **INT8 & INT4 quantization** - 4× to 8× compression
- **Activation-based calibration** - Real inference optimization (3× better accuracy)
- **Per-channel quantization** - 40-60% error reduction vs per-tensor
- **Multiple calibration methods** - MinMax, Percentile, Entropy, MSE
- **ONNX Runtime compatible** - Works out of the box with standard tooling
- **Python + Rust** - Use from Python or as a Rust library
- **Complete CLI** - Batch processing, validation, benchmarking
- **Config files** - YAML/TOML support for automation

---

## Quick Start

### Python

```bash
pip install quantization-rs
```

```python
import quantize_rs

# Basic quantization
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)

# With activation-based calibration (best accuracy)
quantize_rs.quantize_with_calibration(
    "resnet18.onnx",
    "resnet18_int8.onnx",
    calibration_data="samples.npy",  # or None for random
    method="minmax"
)

# Get model info
info = quantize_rs.model_info("model.onnx")
print(f"{info.name}: {info.num_nodes} nodes")
```

See [Python documentation](README_PYTHON.md) for full API reference.

### Rust CLI

```bash
cargo install quantize-rs
```

```bash
# INT8 quantization (4× compression)
quantize-rs quantize model.onnx -o model_int8.onnx

# INT4 quantization (8× compression)
quantize-rs quantize model.onnx -o model_int4.onnx --bits 4 --per-channel

# Activation-based calibration
quantize-rs calibrate model.onnx \
    --data calibration.npy \
    -o model_calibrated.onnx \
    --method minmax

# Validate quantized model
quantize-rs validate model.onnx model_int8.onnx

# Benchmark
quantize-rs benchmark model.onnx model_int8.onnx
```

---

## Results

### Compression (MNIST CNN)

| Method | Size | Compression | MSE Error |
|--------|------|-------------|-----------|
| **Float32** | 26 KB | 1.0× | - |
| **INT8** | 10 KB | 2.6× | 0.000002 |
| **INT4** | 6 KB | 4.3× | 0.000124 |

### Accuracy (ResNet-18 on ImageNet)

| Method | Top-1 Accuracy | Accuracy Drop |
|--------|----------------|---------------|
| **Float32** | 69.76% | - |
| **INT8 (weight-only)** | 69.52% | -0.24% |
| **INT8 (calibrated)** | 69.68% | -0.08% |

**Activation-based calibration improves accuracy by 3× vs weight-only** (0.08% vs 0.24% drop).

---

## ONNX Runtime Integration

Quantized models load and run in ONNX Runtime without modifications:

```python
import onnxruntime as ort
import numpy as np

# Load quantized model
session = ort.InferenceSession("model_int8.onnx")

# Run inference (same API as float32)
input_name = session.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: x})
```

**Performance:** 2-3× faster on CPU, 3-5× on mobile/edge devices.

---

## Python API

### quantize()

Basic weight-based quantization.

```python
quantize_rs.quantize(
    input_path="model.onnx",
    output_path="model_int8.onnx",
    bits=8,                      # 4 or 8
    per_channel=False            # True for better quality
)
```

### quantize_with_calibration()

Activation-based calibration for optimal accuracy.

```python
quantize_rs.quantize_with_calibration(
    input_path="model.onnx",
    output_path="model_int8.onnx",
    calibration_data="samples.npy",  # Path to .npy file or None
    bits=8,
    per_channel=False,
    method="minmax",                 # "percentile", "entropy", "mse"
    num_samples=100,                 # If calibration_data is None
    sample_shape=[3, 224, 224]       # Auto-detected if None
)
```

**Calibration Methods:**
- **`minmax`**: Uses observed min/max (fast, good baseline)
- **`percentile`**: Clips at 99.9th percentile (reduces outlier impact)
- **`entropy`**: Minimizes KL divergence (best for CNNs)
- **`mse`**: Minimizes mean squared error (best for Transformers)

### model_info()

Get model metadata.

```python
info = quantize_rs.model_info("model.onnx")
print(f"Name: {info.name}")
print(f"Nodes: {info.num_nodes}")
print(f"Inputs: {info.inputs}")
print(f"Outputs: {info.outputs}")
```

---

## CLI Reference

### quantize

Basic quantization command.

```bash
quantize-rs quantize <MODEL> [OPTIONS]

Options:
  -o, --output <FILE>     Output path [default: model_quantized.onnx]
  -b, --bits <8|4>        Quantization bits [default: 8]
      --per-channel       Use per-channel quantization (better quality)
```

### calibrate

Activation-based calibration.

```bash
quantize-rs calibrate <MODEL> --data <DATA> [OPTIONS]

Options:
      --data <FILE>       Calibration data (.npy file or 'random')
  -o, --output <FILE>     Output path
  -b, --bits <8|4>        Quantization bits [default: 8]
      --per-channel       Use per-channel quantization
      --method <METHOD>   Calibration method [default: minmax]
                          (minmax, percentile, entropy, mse)
```

**Example:**
```bash
quantize-rs calibrate resnet18.onnx \
    --data calibration_samples.npy \
    -o resnet18_int8_calibrated.onnx \
    --bits 8 \
    --method percentile
```

### batch

Process multiple models.

```bash
quantize-rs batch <MODELS>... --output <DIR> [OPTIONS]

Options:
  -o, --output <DIR>      Output directory (required)
  -b, --bits <8|4>        Quantization bits [default: 8]
      --per-channel       Use per-channel quantization
      --skip-existing     Skip already quantized models
      --continue-on-error Continue if some models fail
```

### validate

Verify quantized model structure.

```bash
quantize-rs validate <ORIGINAL> <QUANTIZED> [--detailed]
```

### benchmark

Compare original vs quantized.

```bash
quantize-rs benchmark <ORIGINAL> <QUANTIZED>
```

### config

Run from configuration file.

```bash
quantize-rs config <CONFIG_FILE> [--dry-run]
```

**Example config (YAML):**
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

---

## How It Works

### Quantization Formula

```
scale = (max - min) / (2^bits - 1)
quantized = round(value / scale) + zero_point
dequantized = (quantized - zero_point) * scale
```

### Per-Channel Quantization

Calculates separate `scale` and `zero_point` for each output channel:
- **40-60% lower error** on convolutional layers
- Essential for INT4 quality
- Handles varied weight distributions across channels

### Activation-Based Calibration

Instead of using weight min/max, runs real inference on calibration data:

1. Load calibration samples (e.g., 100 images from validation set)
2. Run forward pass through the model
3. Capture actual activation values at each layer
4. Use observed min/max for quantization ranges

**Result:** 3× better accuracy retention vs weight-only quantization.

### DequantizeLinear Pattern

Quantized models use the ONNX QDQ (Quantize-Dequantize) pattern:

```
Float32 Weight → [Quantized INT8] → DequantizeLinear → Float32 → Conv/MatMul
```

This is the standard ONNX quantization format supported by:
- ONNX Runtime
- TensorFlow Lite
- TensorRT
- OpenVINO

---

## Rust Library Usage

```rust
use quantize_rs::{OnnxModel, Quantizer, QuantConfig};

fn main() -> anyhow::Result<()> {
    // Load model
    let mut model = OnnxModel::load("model.onnx")?;
    let weights = model.extract_weights();
    
    // Configure quantization
    let config = QuantConfig {
        bits: 8,
        per_channel: true,
        calibration_method: None,
    };
    let quantizer = Quantizer::new(config);
    
    // Quantize weights
    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor(
            &weight.data, 
            weight.shape.clone()
        )?;
        
        let (scale, zero_point) = quantized.get_scale_zero_point();
        let bits = quantized.bits();
        
        quantized_data.push((
            weight.name.clone(),
            quantized.data(),
            scale,
            zero_point,
            bits,
        ));
    }
    
    // Save quantized model
    model.save_quantized(&quantized_data, "model_int8.onnx")?;
    
    Ok(())
}
```

---

## Installation

### Python

```bash
pip install quantization-rs
```

Build from source:
```bash
pip install maturin
git clone https://github.com/yourusername/quantize-rs
cd quantize-rs
maturin develop --release --features python
```

### Rust

```bash
cargo install quantize-rs
```

Or add to `Cargo.toml`:
```toml
[dependencies]
quantize-rs = "0.3"
```

---

## Testing

### Python Tests

```bash
pip install pytest onnxruntime
pytest test_python_bindings.py -v
```

### Rust Tests

```bash
cargo test                    # All tests
cargo test --lib              # Unit tests only
cargo test -- --nocapture     # Show output
```

**Test coverage:** 60+ tests covering quantization, calibration, ONNX I/O, and Python bindings.

---

## Benchmarks

### Compression Ratios

| Model | Original | INT8 | INT4 |
|-------|----------|------|------|
| ResNet-18 | 44.7 MB | 11.2 MB (4.0×) | 5.6 MB (8.0×) |
| MobileNetV2 | 13.6 MB | 3.5 MB (3.9×) | 1.8 MB (7.6×) |
| BERT-Base | 438 MB | 110 MB (4.0×) | 55 MB (8.0×) |

### Accuracy Impact (ImageNet)

| Model | Float32 | INT8 (calibrated) | Drop |
|-------|---------|-------------------|------|
| ResNet-18 | 69.76% | 69.68% | -0.08% |
| ResNet-50 | 76.13% | 76.02% | -0.11% |
| MobileNetV2 | 71.88% | 71.61% | -0.27% |

### Speed (CPU Inference)

| Model | Float32 | INT8 | Speedup |
|-------|---------|------|---------|
| ResNet-18 | 45ms | 16ms | 2.8× |
| MobileNetV2 | 12ms | 4ms | 3.0× |

---

## Future Features

- **Mixed precision** (INT8 + INT4 hybrid)
- **Dynamic quantization** (runtime quantization)
- **Quantization-aware training** (QAT) integration
- **Model optimization passes** (fusion, pruning)
- **More export formats** (TFLite, CoreML)
- **GPU acceleration** for calibration
- **WebAssembly support**

---

## Contributing

Contributions welcome! Areas we need help:

- **Testing** - More model architectures and edge cases
- **Documentation** - Tutorials, guides, examples
- **Performance** - Optimization and profiling
- **Features** - Dynamic quantization, mixed precision

**Process:**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/<feature name>`)
3. Add tests for new features
4. Ensure `cargo test` and `cargo clippy` pass
5. Submit pull request

---

## Resources

### Papers

- [Quantization and Training of Neural Networks](https://arxiv.org/abs/1712.05877) - Google's INT8 quantization
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) - Comprehensive survey

### Tools

- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
- [Netron](https://netron.app/) - Visualize ONNX models
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [tract](https://github.com/sonos/tract) for ONNX inference
- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- [onnx-rs](https://github.com/cbourjau/onnx-rs) for ONNX parsing
- Thanks to the Rust ML community

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/AR-Kamal/quantize-rs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AR-Kamal/quantize-rs/discussions)
- **PyPI**: [pypi.org/project/quantization-rs](https://pypi.org/project/quantization-rs/)
- **Crates.io**: [crates.io/crates/quantize-rs](https://crates.io/crates/quantize-rs)