# Quantify

> Simple neural network quantization toolkit in Rust

🚧 **Work in Progress** - Currently building MVP

## Roadmap

- [ ] Week 1: ONNX loading and basic INT8 quantization
- [ ] Week 2: Calibration and per-channel quantization
- [ ] Week 3: CLI polish and documentation
- [ ] Week 4: Publish to crates.io

## Quick Start
```bash
# Clone and build
git clone https://github.com/AR-Kamal/quantify
cd quantify
cargo build --release

# Quantize a model (coming soon)
./target/release/quantify quantize model.onnx -o model_int8.onnx
```

## Goals

- ✅ Simple CLI interface
- ✅ Pure Rust (no Python dependencies)
- ✅ ONNX format support
- ✅ INT8/INT4 quantization
- ✅ Minimal accuracy loss (<2%)

## License

MIT