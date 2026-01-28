# quantize-rs

> Production-grade neural network quantization toolkit in pure Rust

[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![Downloads](https://img.shields.io/crates/d/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**quantize-rs** reduces neural network size by up to **8x** while preserving accuracy. Convert float32 weights to INT8/INT4 with advanced per-channel quantization and custom packed storage.

> NOTE: THE LATEST UPDATE IS NOT PUBLISHED YET, THE CURRENT VERSION OF quantize-rs STILL 0.1.0

---

## Features

- **INT8 & INT4 quantization** - 4x to 8x compression
- **Per-channel quantization** - 40-60% error reduction vs per-tensor
- **Custom packed storage** - True 8x compression for INT4
- **Fast** - Pure Rust, no Python dependency
- **Complete CLI** - Batch processing, validation, benchmarking
- **ONNX format** - Works with PyTorch, TensorFlow, etc.
- **Config files** - YAML/TOML support for automation

---

## Quick Start

### Installation
```bash
cargo install quantize-rs
```

Or build from source:
```bash
git clone https://github.com/yourusername/quantize-rs
cd quantize-rs
cargo build --release
```

### Basic Usage
```bash
# INT8 quantization (4x compression)
quantize-rs quantize model.onnx -o model_int8.onnx

# INT4 quantization (8x compression)
quantize-rs quantize model.onnx -o model_int4.onnx --bits 4

# Per-channel for better quality
quantize-rs quantize model.onnx -o model_int8.onnx --per-channel

# Validate quantized model
quantize-rs validate model.onnx model_int8.onnx

# Compare performance
quantize-rs benchmark model.onnx model_int8.onnx
```

---

## Results

### ResNet-18 Compression

| Method | Size | Compression | Avg MSE | Notes |
|--------|------|-------------|---------|-------|
| **Original** | 44.65 MB | 1.0x | - | Float32 |
| **INT8** | 11.18 MB | 4.0x | 0.000003 | Standard |
| **INT8 Per-Channel** | 11.18 MB | 4.0x | 0.000002 | 33% better |
| **INT4** | 5.60 MB | 8.0x | 0.000907 | High compression |
| **INT4 Per-Channel** | 5.60 MB | 8.0x | 0.000862 | 5% better |

**Real file sizes achieved with custom packed storage format!**

---

## Documentation

### Commands

#### `quantize` - Quantize a model
```bash
quantize-rs quantize <MODEL> [OPTIONS]

Options:
  -o, --output <FILE>     Output path[default: model_quantized.onnx]
  -b, --bits <8|4>        Quantization bits [default: 8]
      --per-channel       Use per-channel quantization (better quality)
  -h, --help              Print help
```

**Examples:**
```bash
# Basic INT8
quantize-rs quantize resnet18.onnx -o resnet18_int8.onnx

# INT4 with per-channel (best compression + quality)
quantize-rs quantize resnet18.onnx -o resnet18_int4.onnx --bits 4 --per-channel
```

---

#### `batch` - Process multiple models
```bash
quantize-rs batch <MODELS>... --output <DIR> [OPTIONS]

Options:
  -o, --output <DIR>      Output directory (required)
  -b, --bits <8|4>        Quantization bits [default: 8]
      --per-channel       Use per-channel quantization
      --skip-existing     Skip already quantized models
      --continue-on-error Continue if some models fail
```

**Example:**
```bash
quantize-rs batch models/*.onnx --output quantized/ --bits 4 --per-channel
```

---

#### `validate` - Verify quantized model
```bash
quantize-rs validate <ORIGINAL> <QUANTIZED> [OPTIONS]

Options:
      --detailed          Show per-layer analysis
```

**Example output:**
```
✓ Structure Validation
  ✓ Node count matches: 69
  ✓ Input count matches: 9
  ✓ Output count matches: 1

✓ Weight Validation
  ✓ Weight tensor count matches: 102
  ✓ All weight shapes match
  ✓ No numerical issues detected

✓ Size Analysis
  Original:  44.65 MB
  Quantized: 11.18 MB
  Reduction: 75.0% (4.00x smaller)

✓ VALIDATION PASSED
```

---

#### `benchmark` - Compare models
```bash
quantize-rs benchmark <ORIGINAL> <QUANTIZED>
```

---

#### `config` - Run from config file
```bash
quantize-rs config <CONFIG_FILE> [--dry-run]
```

**Example config (YAML):**
```yaml
bits: 4
per_channel: true

models:
  - input: models/resnet18.onnx
    output: quantized/resnet18_int4.onnx
  
  - input: models/mobilenet.onnx
    output: quantized/mobilenet_int4.onnx

batch:
  input_dir: "models/*.onnx"
  output_dir: quantized/
  skip_existing: true
```

---

## How It Works

### Quantization Methods

#### **Per-Tensor Quantization**
Uses global min/max for entire tensor:
```
scale = (max - min) / 255
quantized = round(value / scale) + zero_point
```

#### **Per-Channel Quantization**
Calculates separate scale/zero-point for each output channel:
- **40-60% lower error** on Conv layers
- Essential for INT4 quality
- Handles varied weight distributions

#### **INT4 Bit Packing**
Stores 2 INT4 values per byte:
```
Byte: [AAAA BBBB]
      ↑    ↑
      val1 val2
```
True 8x compression with custom storage format!

---

## Library Usage
```rust
use quantize_rs::{OnnxModel, Quantizer, QuantConfig};

fn main() -> anyhow::Result<()> {
    // Load model
    let mut model = OnnxModel::load("model.onnx")?;
    let weights = model.extract_weights();
    
    // Configure quantization
    let config = QuantConfig {
        bits: 4,                // INT4 for 8x compression
        per_channel: true,      // Better quality
    };
    let quantizer = Quantizer::new(config);
    
    // Quantize each weight
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
    model.save_quantized(&quantized_data, "model_int4.onnx")?;
    
    Ok(())
}
```

---

## Testing

### Download test models
```bash
bash scripts/download_test_models.sh
```

### Test Coverage
```bash
cargo test                    # Run all tests (30+ tests)
cargo test --lib              # Unit tests only
cargo test -- --nocapture     # Show output
```

### Real Model Tests
```bash
# Test on real models (requires ONNX files)
cargo test test_int4_real_model -- --ignored --nocapture
```

**Tested on:**
- ✅ ResNet-18 (44.65 MB → 5.60 MB)
- ✅ MNIST CNN (26 KB → 5.6 KB)
- ✅ MobileNet (13.4 MB → 3.4 MB)

---

## To-Do
- **Calibration datasets** - Use sample data for optimal quantization ranges
  - Percentile-based range selection
  - Entropy minimization
  - MSE optimization
  - Industry-standard approach for production quantization

---

## Future Features
- Mixed precision (INT8 + INT4)
- Dynamic quantization (runtime)
- Quantization-aware training (QAT) support
- Model optimization passes (fusion, pruning)
- WebAssembly support
- Python bindings
- More export formats (TFLite, CoreML)

---

## Contributing

Contributions are welcome! Areas we'd love help with:
- **Calibration datasets** - Sample data loading and statistics
- **More quantization methods** - Dynamic, mixed-precision
- **Testing** - More model formats and edge cases
- **Documentation** - Tutorials, guides, use cases
- **Performance** - Optimization and benchmarking

**Process:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/<feature's name>`)
3. Add tests for new features
4. Ensure `cargo test` and `cargo clippy` pass
5. Submit a pull request

---

## Resources

### Papers & References
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) - Google's INT8 quantization
- [LUT-NN: Empower Efficient Neural Network Inference with Centroid Learning and Table Lookup](https://arxiv.org/abs/1911.02929) - INT4 techniques
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) - Comprehensive overview

### Tools & Frameworks
- [ONNX](https://onnx.ai/) - Open Neural Network Exchange
- [TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quantization) - Mobile quantization
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) - PyTorch approach

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [onnx-rs](https://github.com/cbourjau/onnx-rs) for ONNX parsing
- Inspired by TensorFlow Lite and PyTorch quantization
- Thanks to the Rust ML community for feedback and support

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/AR-Kamal/quantize-rs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AR-Kamal/quantize-rs/discussions)
- **Author**: [@AR-Kamal](https://github.com/AR-Kamal)
