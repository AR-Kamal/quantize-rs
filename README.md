# quantize-rs

**ONNX model quantization for Rust, Python, and the command line.**

[![CI](https://github.com/AR-Kamal/quantize-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AR-Kamal/quantize-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![PyPI](https://img.shields.io/pypi/v/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

quantize-rs converts selected FP32 ONNX weights to INT8 or INT4 and writes
ONNX QuantizeLinear/DequantizeLinear (QDQ) graphs. It also supports static INT8
activation calibration for Conv models and an experimental MatMul/Gemm path.
Models can be processed through the CLI, Rust library, or Python bindings.

Smaller model files do not guarantee faster inference. Accuracy, execution
precision, and memory use depend on the model, runtime, and hardware. See the
[recorded evaluations](#evaluation-results) for measured results and limitations.

These examples describe the **0.10 API**. See the [changelog](CHANGELOG.md) for
version history and migration notes.

## Supported workflows

| Workflow | What is quantized | Entry points | Requirements |
|----------|-------------------|--------------|--------------|
| Weight-only INT8 / INT4 | Direct Conv, MatMul, and Gemm weight initializers | CLI `quantize` / `batch`, Python `quantize`, Rust `Quantizer` | Inline FP32 weights; no calibration dataset |
| Static Conv INT8 | Selected Conv weights and activation boundaries | CLI `calibrate`, Python `quantize_with_calibration`, Rust `quantize_static` | Representative data, opset >= 13, one fixed FP32 NCHW input with batch 1 |
| Experimental matrix INT8 | Selected MatMul/Gemm weights and activation boundaries | CLI `calibrate-matrix`, Rust `quantize_static_matrix` | Restricted graphs, representative data, one fixed FP32 `[1,K]` input |

Weight quantization supports per-tensor or per-channel scales, symmetric or
asymmetric ranges, initializer exclusions, size thresholds, and per-layer bit
widths. Per-channel axes follow the operator: Conv axis 0, MatMul's last weight
axis, and Gemm axis 1 or 0 according to `transB`.

INT4 values use INT8 storage by default. Enable `--native-int4` or
`native_int4=True` to pack two weights per byte in ONNX INT4 tensors; this requires
runtime support for INT4 DequantizeLinear at opset 21. Total file size includes
scales, zero points, graph metadata, and tensors that remain FP32.

## Installation

### Command line

Requires Rust **1.88+**:

```bash
cargo install quantize-rs --locked
```

### Python

Requires Python **3.9+**:

```bash
python -m pip install quantization-rs
```

The package name is `quantization-rs`; the Python import is `quantize_rs`.
See the [Python API reference](README_PYTHON.md) for installation and usage details.

### Rust library

```toml
[dependencies]
quantize-rs = "0.10"
```

The `calibration` feature is enabled by default. Disable default features for a
weight-only build. Optional features include `python`, `mmap`, and
`safetensors-input`. Memory mapping avoids an intermediate file-read buffer;
model decoding still allocates tensors.

### Build this checkout

```bash
git clone https://github.com/AR-Kamal/quantize-rs.git
cd quantize-rs
cargo build --locked --release --bin quantize-rs
```

The executable is `target/release/quantize-rs` (`quantize-rs.exe` on Windows).
Registry commands install a published package; building the checkout uses its
current source.

For Python development, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install maturin
maturin develop --locked --release --features python
```

## Quick start

### CLI

```bash
# Quantize eligible weights to symmetric per-channel INT8.
quantize-rs quantize model.onnx -o model_int8.onnx --per-channel --symmetric

# Use packed native INT4 weights.
quantize-rs quantize model.onnx -o model_int4.onnx --bits 4 --per-channel --native-int4

# Check graph structure, connectivity, and weight reconstruction errors.
quantize-rs validate model.onnx model_int8.onnx --detailed

# Inspect model metadata as JSON.
quantize-rs info model_int8.onnx --format json
```

### Python

```python
import quantize_rs

quantize_rs.quantize(
    "model.onnx", "model_int8.onnx",
    per_channel=True, symmetric=True,
)

quantize_rs.quantize(
    "model.onnx", "model_int4.onnx",
    bits=4, per_channel=True, native_int4=True,
)

info = quantize_rs.model_info("model_int8.onnx")
print(f"{info.name}: {info.num_nodes} nodes")
```

### Rust

```rust
use quantize_rs::{OnnxModel, QuantConfig, QuantizeError, Quantizer};

fn main() -> Result<(), QuantizeError> {
    let mut model = OnnxModel::load("model.onnx")?;
    let config = QuantConfig::int8().with_per_channel(true).with_symmetric(true);
    let outputs = Quantizer::new(config).quantize_model(&model)?;
    let weights = outputs.into_iter().map(|output| output.qdq).collect::<Vec<_>>();
    model.save_quantized(&weights, "model_int8.onnx")?;
    Ok(())
}
```

Use `SaveOptions::default().with_native_int4(true)` with
`save_quantized_with_options` when saving INT4 results in native storage. See the
[Rust API documentation](https://docs.rs/quantize-rs) and [examples](examples/README.md).

## Activation calibration

For a supported Conv model, prepare representative preprocessed FP32 samples in
an NPY file shaped `[samples,C,H,W]`. The model input must be fixed `[1,C,H,W]`.
Measure accuracy on a separate held-out dataset.

```bash
quantize-rs calibrate cnn.onnx --data samples.npy -o cnn_int8.onnx --method minmax --per-channel --symmetric
```

The equivalent Python call is:

```python
import quantize_rs

quantize_rs.quantize_with_calibration(
    "cnn.onnx", "cnn_int8.onnx", calibration_data="samples.npy",
    method="minmax", per_channel=True, symmetric=True,
)
```

Methods include `minmax`, `percentile`, `percentile:NN`, `entropy`, and `mse`.
Activation ranges come from real inference; weight ranges come from the weights.
The `symmetric` setting applies to weights. Activations use scalar asymmetric
INT8 parameters. Missing data and unsupported layouts return errors.

The **0.10 calibration behavior differs from 0.9**: it inserts activation QDQ and
requires representative data. See [the calibration contract and migration](CALIBRATION.md).
INT4 and mixed precision belong to the weight-only path.

For restricted matrix graphs, use `calibrate-matrix` with FP32 data shaped
`[samples,K]`. This experimental Rust/CLI path does not calibrate token-based
Transformer models and is not exposed through Python. See [matrix calibration](MATRIX_CALIBRATION.md).

## Batch processing and CLI tools

```bash
# Process matching files with four parallel workers.
quantize-rs batch "models/*.onnx" -o quantized/ --per-channel --symmetric --jobs 4

# Keep an initializer in FP32 and override another to INT4.
quantize-rs quantize model.onnx -o mixed.onnx --exclude sensitive.weight --layer-bits projection.weight=4

# Compare file sizes and model structure.
quantize-rs benchmark model.onnx model_int8.onnx --format json

# Preview a YAML or TOML quantization configuration.
quantize-rs config quantize.yaml --dry-run
```

Selection flags use **initializer names**, which can differ from node names.
`--min-elements` skips small weights. The `validate`, `info`, and `benchmark`
commands support `--format json`; `validate` exits nonzero when its checks fail.

`benchmark` reports file size, compression, and structural statistics. It does
**not measure inference latency**. Likewise, `validate` checks the export and
weight reconstruction, not task accuracy. Use runtime evaluation on your own data
before choosing a quantization configuration.

Run `quantize-rs --help` or `quantize-rs <command> --help` for all options. See the
[YAML](examples/config.yaml) and [TOML](examples/config.toml) examples for
configuration-file workflows.

## Runtime compatibility and limitations

- **Selected weights only.** Embeddings, biases, unrelated constants, and weights
  reached indirectly through Reshape/Transpose stay FP32. Conflicting shared
  uses are rejected. Subgraph capture analysis is unsupported. See
  [operator selection and axes](OPERATOR_QUANTIZATION.md).
- **Inline ONNX tensors.** External tensor data files are unsupported. Export
  weights inline where model size permits. PyTorch and TensorFlow models must
  first be exported to ONNX.
- **Restricted calibration.** Static paths reject unsupported input layouts,
  existing QDQ, custom domains, subgraphs, external tensors, and ONNX sections
  that cannot be preserved. tract's operator support also limits calibration.
- **ONNX round-trip limits.** Weight-only saving does not preserve local functions,
  sparse initializers, or training information; loading emits a warning when
  these sections are present. Outputs that depend on them may be invalid.
- **Static INT8 on CPU.** Configure ORT with `session.x64quantprecision=1` to
  avoid integer saturation on x64 CPUs without VNNI. See
  [CPU runtime precision](CALIBRATION.md#cpu-runtime-precision).
- **Runtime-dependent precision.** Weight-only output inserts DequantizeLinear
  nodes without explicitly quantizing activations. A runtime may use FP32 kernels
  or fuse the graph into quantized kernels. Optimizations can add rounding error;
  [operator validation](OPERATOR_QUANTIZATION.md#validation) explains the separate
  FP32-activation parity and default-runtime quality checks.
- **INT4 needs accuracy evaluation.** Per-tensor INT4 can severely degrade models
  whose channels have different weight ranges. Per-channel quantization can help,
  but neither bit width guarantees a particular accuracy or speedup.

## Evaluation results

These are recorded CPU measurements with **ONNX Runtime 1.22.1**, not predictions
for other hardware, runtime versions, or models. Both quantized runs were slower
than their FP32 baselines on the measured machine.

| Evaluation | Recorded quality | Model file size | Full report |
|------------|------------------|-----------------|-------------|
| MNIST CNN, calibrated Conv INT8 | 98.91% top-1 vs 98.90% FP32 on 10,000 test images | 18,172 vs 26,143 bytes | [CNN methodology and results](eval/CNN_EVALUATION.md) |
| GPT-2, symmetric per-channel weight INT8 | 37.2634 perplexity vs 36.6704 FP32 on a fixed 32,768-token prefix | 56.68% smaller | [GPT-2 coverage and results](eval/GPT2_EVALUATION.md) |

The reports include dataset scope, model hashes, runtime settings, latency,
process memory, and reproduction commands. They establish bounded regression
results rather than general production accuracy or acceleration claims.

## Documentation

- [Python API reference](README_PYTHON.md)
- [Operator selection, axes, and compatibility](OPERATOR_QUANTIZATION.md)
- [Static Conv calibration and migration](CALIBRATION.md)
- [Experimental matrix activation calibration](MATRIX_CALIBRATION.md)
- [Rust examples and configurations](examples/README.md)
- [Changelog](CHANGELOG.md)

## Development and contributing

```bash
cargo test --locked --all-features
cargo clippy --locked --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
```

For runtime checks, install this checkout's Python bindings in a virtual
environment as described above, then:

```bash
python -m pip install numpy onnx onnxruntime
cargo build --locked --bin quantize-rs
python -m unittest discover -s eval -p "test_*.py" -v
python eval/operator_axes_smoke_test.py --binary target/debug/quantize-rs --python
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs --python
python eval/matrix_calibration_smoke_test.py --binary target/debug/quantize-rs
```

Use `.exe` binary paths on Windows. Criterion benchmarks live in `benches/`;
see [fuzzing instructions](fuzz/README.md) for the ONNX loader target.

Contributions are welcome through pull requests. Include a reproducible example
for bug reports, tests for behavior changes, and updates to the relevant docs.

## License

[MIT](LICENSE). Built with [tract](https://github.com/sonos/tract),
[PyO3](https://github.com/PyO3/pyo3), and
[prost](https://github.com/tokio-rs/prost).
