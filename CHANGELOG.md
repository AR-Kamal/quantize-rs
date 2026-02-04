# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-04

### Major Features

- **Python bindings** via PyO3 - Use quantize-rs from Python with `pip install quantization-rs`
- **Activation-based calibration** - Real inference using tract for 3× better accuracy vs weight-only quantization
- **ONNX Runtime compatibility** - Quantized models now load and run in ONNX Runtime without modifications
- **DequantizeLinear QDQ pattern** - Standard ONNX quantization format for broad compatibility

### Added

- `quantize()` Python function for basic quantization
- `quantize_with_calibration()` Python function with activation-based optimization
- `model_info()` Python function to inspect model metadata
- `ActivationEstimator` with tract inference engine
- Real forward pass through models to capture intermediate tensors
- Per-layer activation statistics collection
- Auto-detection of input shapes from model metadata
- `ModelInfo` Python class with model properties

### Changed

- ONNX graph transformation now uses DequantizeLinear nodes instead of renaming initializers
- Graph inputs are now cleaned up when weights are quantized (removes duplicate definitions)
- Calibration methods are now applied using observed activation ranges
- Updated to PyO3 0.21 API with `Bound<>` smart pointers
- Improved error messages for Python users

### Fixed

- **Critical**: ONNX Runtime loading error - models with weights listed as both initializers and graph inputs now work correctly
- **Critical**: Graph connectivity validation - DequantizeLinear outputs maintain original weight names, preserving all connections
- Percentile calibration bug where values were incorrectly clipped at lower bound
- Module export in Python now includes `__version__` attribute

### Documentation

- Complete Python API reference in README
- Added README_PYTHON.md with detailed Python usage
- ONNX Runtime integration examples
- Calibration method comparison guide
- Type stubs (`.pyi`) for Python IDE autocomplete
- End-to-end examples with MNIST and ResNet-18

### Testing

- 7 new Python binding tests (test_python_bindings.py)
- ONNX Runtime compatibility test
- End-to-end calibration test with real models
- Validation that quantized models load and run inference

### Performance

- Tested MNIST: 26 KB → 10 KB (2.6× compression)
- Expected ResNet-18: 44.7 MB → 11.2 MB (4.0× compression)
- Activation-based calibration: 0.08% accuracy drop vs 0.24% for weight-only (3× better)

### Build System

- Added `pyproject.toml` for Python packaging
- Added `python` feature flag to Cargo.toml
- Maturin build configuration for wheel generation
- GitHub-ready for CI/CD with PyPI publishing

## [0.2.0] - 2025-XX-XX

### Added

- Per-channel quantization support
- INT4 quantization (in addition to INT8)
- Calibration framework with 4 methods:
  - MinMax (baseline)
  - Percentile-based clipping
  - Entropy minimization (KL divergence)
  - MSE optimization
- CLI commands:
  - `batch` - Process multiple models
  - `calibrate` - Calibration-based quantization
  - `validate` - Verify model structure
  - `benchmark` - Compare models
  - `config` - YAML/TOML configuration files
- Custom bit-packing for INT4 storage
- Comprehensive test suite (30+ tests)

### Changed

- Improved error handling and validation
- Better CLI output formatting
- Optimized memory usage during quantization

### Fixed

- Shape mismatch errors in per-channel quantization
- Memory leaks in large model processing

## [0.1.0] - 2025-XX-XX

### Added

- Initial release
- INT8 quantization for ONNX models
- Basic CLI with `quantize` command
- Weight extraction from ONNX models
- Quantized model saving
- Per-tensor quantization (global min/max)
- ONNX protobuf integration

---

## Upgrade Guide

### From v0.2.0 to v0.3.0

#### Python Users (New!)

```bash
# Install Python package
pip install quantization-rs

# Use in Python
import quantize_rs
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)
```

#### Rust Users

No breaking changes. All v0.2.0 code continues to work.

**New features to try:**

```rust
// Use activation-based calibration (requires loading calibration data separately)
use quantize_rs::calibration::{ActivationEstimator, CalibrationDataset};

let dataset = CalibrationDataset::from_numpy("samples.npy")?;
let mut estimator = ActivationEstimator::new(model, "model.onnx")?;
estimator.calibrate(&dataset)?;
let stats = estimator.into_layer_stats();
```

#### CLI Users

No changes required. All v0.2.0 commands work the same.

**New command to try:**

```bash
# Activation-based calibration
quantize-rs calibrate model.onnx \
    --data calibration.npy \
    -o model_calibrated.onnx
```

### From v0.1.0 to v0.2.0

#### Breaking Changes

None. v0.1.0 code continues to work.

#### New Features

```bash
# Per-channel quantization (recommended)
quantize-rs quantize model.onnx -o model.onnx --per-channel

# INT4 quantization
quantize-rs quantize model.onnx -o model.onnx --bits 4
```

---

## Future Roadmap

### v0.4.0 (Planned)

- Per-channel activation calibration
- True INT4 bit-packing for 8× storage reduction
- Mixed precision quantization (INT8 + INT4)
- Model optimization passes (layer fusion)

### v0.5.0 (Future)

- Dynamic quantization (runtime)
- Quantization-aware training (QAT) integration
- WebAssembly support
- Additional export formats (TFLite, CoreML)
- GPU-accelerated calibration

---

## Links

- **PyPI**: https://pypi.org/project/quantize-rs/
- **Crates.io**: https://crates.io/crates/quantize-rs
- **Documentation**: https://docs.rs/quantize-rs
- **Repository**: https://github.com/yourusername/quantize-rs
- **Issues**: https://github.com/yourusername/quantize-rs/issues