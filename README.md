# quantize-rs

Neural network quantization toolkit for ONNX models, written in Rust with Python bindings.

This checkout prepares **v0.10.0**; publication is pending the
[release checklist](RELEASE_v0.10.0.md). Registry installation commands below
install the published version; build this checkout to test the candidate.

[![Crates.io](https://img.shields.io/crates/v/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Crates.io Downloads](https://img.shields.io/crates/d/quantize-rs.svg)](https://crates.io/crates/quantize-rs)
[![Documentation](https://docs.rs/quantize-rs/badge.svg)](https://docs.rs/quantize-rs)
[![PyPI](https://img.shields.io/pypi/v/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/quantization-rs.svg)](https://pypi.org/project/quantization-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

quantize-rs converts float32 ONNX models to INT8 or INT4 representation using post-training quantization. It supports weight-only quantization and a restricted static INT8 Conv activation path, with per-channel weights and standard ONNX QDQ output. See [calibration behavior and migration](CALIBRATION.md) for the corrected calibration contract.

> **Scope:** weight-only quantization selects direct inline FP32 Conv/MatMul/Gemm weights with operator-specific axes. See [selection rules and migration](OPERATOR_QUANTIZATION.md). A [bounded GPT-2 INT8 evaluation](eval/GPT2_EVALUATION.md) verifies matrix-weight coverage and held-out perplexity; it is smaller but slower on the measured CPU. Historical weight-only evaluation also covers CNNs. Static calibration supports selected Conv layers with one fixed FP32 NCHW input and batch 1. It has synthetic regressions and a [labeled MNIST CNN evaluation](eval/CNN_EVALUATION.md) covering accuracy, latency and process memory. Broader deployment validation is required before broader production accuracy claims. See [CALIBRATION.md](CALIBRATION.md).

## Features

An experimental [matrix activation-calibration path](MATRIX_CALIBRATION.md) is now
available through `calibrate-matrix` and Rust for fixed FP32 `[1,K]` inputs. It
demonstrates integer MatMul/Gemm execution on simple graphs; token-model calibration
and shared-edge fusion require further work.

- **INT8 and INT4 quantization** -- per-tensor or per-channel, asymmetric or symmetric
- **Native INT4 storage** -- `--native-int4` writes ONNX `DataType::Int4` (opset 21) to pack two quantized weights per byte; total model compression depends on metadata and unchanged tensors
- **Symmetric quantization** -- `--symmetric` forces `zero_point == 0`, required by most ORT / TensorRT INT8 matmul kernels for per-channel weights
- **Static INT8 Conv calibration** -- runs real inference on representative data via [tract](https://github.com/sonos/tract), then inserts activation QuantizeLinear/DequantizeLinear pairs. Weights retain their own ranges. Requires opset >= 13 and one fixed FP32 NCHW input with batch 1.
- **Multiple calibration methods** -- MinMax, Percentile (99.9th), Entropy (KL divergence), MSE -- with histogram-direct range optimization (no sample regeneration)
- **ONNX QDQ output format** -- quantized models use `DequantizeLinear` nodes and load directly in ONNX Runtime
- **Graph connectivity validation** -- verifies that every node input resolves after quantization
- **Per-layer selection** -- exclude layers by name, set per-layer bit widths, or skip small tensors via `min_elements`; available on `quantize` and `batch`; `calibrate` supports Conv exclusions and thresholds, with INT8-only overrides
- **CLI** -- single-model quantization, batch processing, validation, benchmarking, config-file driven workflows; `validate` / `info` / `benchmark` support `--format json` for machine-readable output
- **Parallel batch processing** -- `--jobs N` quantizes multiple models concurrently
- **Optional memory-mapped loading** (`mmap` feature) -- avoids the intermediate file-read buffer; prost still allocates decoded tensors
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
quantize-rs = "0.10"
```

Requires **Rust 1.88+** (MSRV).

## Quick start

### Python

```python
import quantize_rs

# Weight-based INT8 quantization
quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)

# INT4 with native opset-21 storage (packed 4-bit weights)
quantize_rs.quantize("model.onnx", "model_int4.onnx", bits=4, native_int4=True)

# Symmetric per-channel INT8 (recommended for ORT/TensorRT matmul kernels)
quantize_rs.quantize("model.onnx", "out.onnx", bits=8, per_channel=True, symmetric=True)

# Static INT8 Conv activation calibration (requires representative data)
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

# INT4 with native opset-21 storage (packed 4-bit weights)
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

    // Select direct Conv/MatMul/Gemm weights, infer their axes, and quantize
    // with filtering and layer bit overrides.
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

```bash
quantize-rs calibrate model.onnx --data samples.npy -o calibrated.onnx \
    --method minmax --per-channel --symmetric
```

Static INT8 quantization of selected Conv weights and their activation boundaries.
Requires opset >= 13, one fixed FP32 NCHW input with batch 1, and representative
`.npy` samples (or `.safetensors` with the optional feature). `--exclude` and
`--min-elements` select Conv weights. `--layer-bits` accepts only `=8` here.
`--bits 4` and `--native-int4` are rejected; use `quantize` for INT4/mixed precision.
Methods: `minmax`, `percentile`, `percentile:NN`, `entropy`, `mse`.
Missing or unsupported data never triggers random fallback. See [CALIBRATION.md](CALIBRATION.md).

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

Checks structure preservation, graph connectivity, weight shapes, and numerical sanity (all-zero detection, constant-value detection). With `--detailed`, prints per-layer error analysis. `--format json` emits a parseable report on stdout (banner suppressed). **Exits non-zero when validation fails**, so it can gate a CI pipeline (the JSON report still carries `validation_passed`).

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

For INT8, the quantized range is [-128, 127]. For INT4, it is [-8, 7]. INT4 values are bit-packed (two values per byte), using four bits per weight before byte padding, scales and zero points. By default they are widened to INT8 bytes on disk (DequantizeLinear requires INT8 input in opsets < 21); pass `--native-int4` (CLI) or `native_int4=True` (Python) to write them as ONNX `DataType::Int4` in opset 21 with two quantized values per byte. Total model size also includes quantization parameters, graph data and unchanged tensors. The runtime must support opset 21 to load the native variant.

### Symmetric vs asymmetric

By default, quantization is asymmetric: `zero_point` is offset to fit the data range. Pass `--symmetric` to force `zero_point == 0` and use a balanced range `[-|max|, +|max|]`. Most ONNX Runtime / TensorRT INT8 matmul kernels require symmetric per-channel quantization for weights, so `--per-channel --symmetric` is the recommended weight configuration when targeting accelerated INT8 inference — the static Conv path additionally inserts activation QDQ, while weight-only output alone does not guarantee accelerated inference (see [Known limitations](#known-limitations)).

### Per-channel quantization

Computes separate scale and zero_point for each output channel: axis 0 for Conv, the last weight dimension for MatMul, and axis 1 for Gemm unless `transB=1`, which uses axis 0. Compatible shared weights are quantized once; conflicting uses return an error. See [operator selection and axes](OPERATOR_QUANTIZATION.md), including the legacy axis-0 tensor APIs.

### Static activation calibration

`calibrate` observes original ONNX tensor edges through tract. Each selected Conv
input and output receives a QuantizeLinear/DequantizeLinear pair with scalar INT8
activation parameters. Weight parameters are calculated independently from the
weights. Activation scales always include zero. The `symmetric` setting applies
to weights; activations use asymmetric INT8 parameters.

Representative calibration inputs are required. Accuracy improvements and speedups
are workload-dependent and must be measured. [CALIBRATION.md](CALIBRATION.md)
describes supported models, selection semantics, and migration from v0.9.

### Weight-only output format

Quantized models use the ONNX QDQ pattern. For each quantized weight, the original float32 initializer is replaced with:

- `{name}_quantized` -- INT8 tensor (same shape)
- `{name}_scale` -- float32 scalar, or vector for per-channel weights
- `{name}_zp` -- matching INT8 (or native INT4) scalar/vector
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
# Rust tests, including real tract calibration regression tests
cargo test

# All optional features (including Python, mmap and safetensors-input)
cargo test --all-features

# With output
cargo test -- --nocapture

# Integration tests requiring model files on disk
cargo test -- --ignored --nocapture

# Criterion benchmarks
cargo bench

# Fuzz the ONNX loader (nightly toolchain required)
cargo +nightly fuzz run onnx_load   # see fuzz/README.md

# Offline evaluation math and coverage regressions
python -m unittest discover -s eval -p test_gpt2_evaluation.py -v

# Python tests (requires maturin develop)
python eval/operator_axes_smoke_test.py --binary target/debug/quantize-rs --python
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs --python
python eval/matrix_calibration_smoke_test.py --binary target/debug/quantize-rs
```

## Known limitations

These limits distinguish weight-only `quantize` from static INT8 `calibrate`.

- **Weight-only quantization targets file size.** `quantize` inserts weight DequantizeLinear nodes; runtime compute and memory may stay FP32. `calibrate` additionally inserts activation QDQ for selected Conv layers, but acceleration depends on runtime fusion and hardware. No latency improvement is claimed by this release.
- **Weight selection targets direct Conv/MatMul/Gemm inputs.** Embeddings, unrelated constants and indirect weights through Reshape/Transpose stay FP32. Selected weights with conflicting axes or unsupported shared uses are rejected. Subgraph capture analysis is not supported. See [selection rules](OPERATOR_QUANTIZATION.md).
- **ONNX input only.** PyTorch and TensorFlow models must be exported to ONNX first.
- **Static calibration is deliberately narrow.** It supports selected inline FP32 Conv weights in graphs at opset >= 13, with one fixed FP32 NCHW input and batch 1. Custom domains, subgraphs, existing QDQ, external tensors and unpreserved ONNX sections are rejected. Other operators remain floating point. Weight-only quantization retains its broader existing scope.
- **Per-channel DequantizeLinear** writes 1-D scale/zero_point tensors with the `axis` attribute. ONNX Runtime supports this in opset >= 13.
- **Native INT4 storage** requires an ONNX runtime with opset 21 support. Without `--native-int4`, INT4 values are widened to INT8 bytes on disk (one byte per quantized weight, versus four bytes for FP32; total model compression varies).
- **Per-tensor INT4 is not usable on depthwise-separable convolutions** (MobileNet-style). With only 16 levels and high per-channel weight variance, per-tensor INT4 weight-only quantization is near-random (MobileNetV2 INT4 ≈ 0.07 cosine vs FP32). Use `quantize --per-channel` and evaluate accuracy; static activation calibration supports INT8 only. INT8 works well (MobileNetV2 INT8 ≈ 0.98, per-channel ≈ 0.997).
- **External-data models are not supported.** quantize-rs reads only inline tensor data, so a model whose weights live in a sidecar `.onnx.data` file (common for exports above ~2 GB) is rejected with a clear error. Re-save with weights embedded first — `onnx.load(path, load_external_data=True)` then `onnx.save(model, out)` without external data.
- **A few ONNX sections are not preserved on save.** The protobuf round-trip drops `ModelProto.functions` (local-function custom ops), `GraphProto.sparse_initializer`, and `training_info`; quantize-rs prints a stderr warning when a loaded model carried any of them. Models that rely on local functions in particular may be invalid after quantization — verify the output before deploying.

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
