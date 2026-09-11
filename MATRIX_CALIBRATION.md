# Experimental MatMul/Gemm activation calibration

`quantize_static_matrix` and CLI `calibrate-matrix` add an explicit, restricted
matrix activation-QDQ path. It uses real tract inference to collect activation
ranges and operator-aware symmetric per-channel INT8 weights. The existing
Conv-only `quantize_static`, `calibrate` and Python calibration entry points
retain their contract.

## Supported contract

- One fixed FP32 input with shape **`[1, K]`**, with positive, representable K.
  Each calibration sample has shape `[K]`; the NPY file has shape `[samples, K]`.
  Nonempty finite representative data is mandatory.
- Standard-domain MatMul/Gemm with direct inline FP32 rank-2 initializer weights.
  The left operand must depend on the model input. Add, Relu and Identity are
  the only additional supported operators in this first prototype.
- Gemm supports `transA` and `transB` values 0 or 1. `alpha` and `beta` must be
  absent or 1. Bias is absent or an inline FP32 output-channel vector **`[N]`**.
  Scalar and matrix bias forms are rejected, including `[1,N]`.
- Symmetric per-channel INT8 weights are required. MatMul uses axis 1; Gemm uses
  axis 1 or axis 0 for `transB=1`. Biases retain their FP32 initializer contents.
- Activations use scalar asymmetric INT8 Q/DQ with zero included in their range.
  MinMax (CLI default), Percentile, Entropy and MSE methods use the existing
  statistics implementation. Activation statistics never determine weight ranges.
- Opset >= 13, inline finite FP32 scalar/vector/matrix initializers, preserved
  ONNX sections, valid graph definitions and supported operator arities/attributes.

Token IDs, dynamic or multiple inputs, other input ranks, batched weight tensors,
indirect weights, custom domains, subgraphs, existing QDQ, external tensor files,
INT4 and mixed precision are outside this prototype. It does not calibrate GPT-2.
Mathematical shape compatibility is also checked by tract before any save.

## Selection, sharing and failures

`excluded_layers` and `min_elements` select weight initializers. Selected weights
must have compatible direct matrix uses; sharing a selected weight with an Add,
bias role or a conflicting axis is rejected. Excluded tensors stay FP32.

Shared activation inputs reuse QDQ. Adjacent selected operators reuse the
upstream output boundary. Selected outputs are quantized for every consumer,
including excluded downstream operators; exclusion does not undo an upstream
activation boundary. Original graph input/output names are preserved.

Activation names avoid collisions. Weight-QDQ collisions and malformed inputs
fail before output replacement. Calibration and graph rewriting operate before
the existing atomic save; failures preserve an existing output file.

## Usage

```bash
quantize-rs calibrate-matrix model.onnx --data samples.npy -o matrix_int8.onnx
quantize-rs calibrate-matrix model.onnx --data samples.npy -o matrix_int8.onnx \
    --method percentile:99.9 --exclude sensitive.weight --min-elements 16
```

The CLI fixes weights to symmetric per-channel INT8; it has no INT4, native-INT4
or asymmetric-weight switch. Requires the `calibration` Cargo feature (enabled
by default).

```rust
use quantize_rs::{quantize_static_matrix, CalibrationDataset, QuantConfig};

fn main() -> Result<(), quantize_rs::QuantizeError> {
    let dataset = CalibrationDataset::from_numpy("samples.npy")?;
    let config = QuantConfig::int8().with_per_channel(true).with_symmetric(true);
    quantize_static_matrix("model.onnx", "matrix_int8.onnx", &dataset, config)?;
    Ok(())
}
```

`OnnxModel::matrix_calibration_sample_shape()` reports `[K]` after validating the
input shape/type. Quantization performs the remaining graph/configuration checks.
No existing public configuration fields or Python argument lists were changed;
the experimental matrix entry point is currently available through Rust and CLI.

## Numerical and runtime validation

The [recorded synthetic run](eval/results/matrix-activation-smoke.json) covers
13 cases on ONNX Runtime 1.22.1 CPUExecutionProvider. All meet the numerical gate;
the largest relative RMSE is 3.14%. Eleven single-layer/serial cases execute
integer matrix kernels. Two fan-out cases retain float Gemm/MatMul execution.
Optimized and unoptimized outputs were identical in this run. These are synthetic
regressions, not representative task-quality or latency measurements.

```bash
cargo test --test matrix_calibration
python eval/matrix_calibration_smoke_test.py --binary target/debug/quantize-rs \
    --report target/matrix-activation-smoke.json
```

Use an `.exe` binary path on Windows. Tests generate their own graphs and data;
no model download is needed. They cover independent weight ranges, calibration
dataset sensitivity, transpose axes, sharing, selection filters, preserved public
outputs/unselected tensors and failures that preserve existing output files.

Cases requiring integer fusion use the shared
[CPU precision setting](CALIBRATION.md#cpu-runtime-precision) to avoid integer
saturation on x64 CPUs without VNNI. JSON reports record `session_config` for
each case. Graph fusion and the integer-kernel requirements remain enabled.

Runtime checks use NumPy seed 917, 256 uniform calibration samples in `[-2,2]`
and 32 separate held-out samples in `[-1,1]`. Each case must have pooled output
relative RMSE below 5% versus FP32. Optimized/unoptimized QDQ results must agree
within one output quantization step plus a small floating-point allowance, since
integer requantization can round ties differently.

For the supported single-layer and serial-chain cases, the test requires actual
`QLinearMatMul`/`QGemm` execution and no remaining float MatMul/Gemm kernels in
the ORT profile. This checks runtime behavior, not merely serialized QDQ nodes.
ORT can internally convert signed activations to UINT8 and prepack weights;
the exported activation zero points remain INT8.

**Shared QDQ edges can block fusion.** Shared-weight and fan-out cases retain
ORT default session settings and report actual kernel types without requiring
fusion. ORT 1.30 precision conversion on AVX2 can fail to initialize a shared-weight
graph with `Attempt to replace the existing tensor`. The default-runtime cases
still enforce the same accuracy and parity limits; they are not evidence of
overflow-safe integer execution. In the AVX2 verification they execute float
MatMul/Gemm kernels. Their report has an empty `session_config` object.
Likewise, constant matrix-shaped Gemm bias blocked fusion in the initial runtime
probe, which is why this prototype accepts vector bias only. Kernel behavior
depends on the runtime/provider/version. Integer execution alone is not evidence
of a latency improvement; no performance claim is made by these small tests.

The design follows [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html),
[MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) and
[ORT QDQ quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
It addresses the next mechanism identified by the
[GPT-2 weight-only evaluation](eval/GPT2_EVALUATION.md). Applying it to token
models still requires a typed-input/backend design and a separate held-out
quality/performance evaluation. Shared-edge fusion also needs further work.
