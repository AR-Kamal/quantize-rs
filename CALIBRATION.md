# Static INT8 Conv calibration

The v0.10.0 candidate replaces v0.9's calibrated-weight behavior with actual
activation QDQ. This is a behavioral migration in a pre-1.0 minor release;
publication is pending the [release gates](RELEASE_v0.10.0.md).

## What changed

Previously the estimator collected node-name statistics while the quantizer looked
them up using initializer names. Missing matches silently used weight ranges;
matches could incorrectly apply activation ranges to weights. Per-channel mode
ignored activation ranges. The old integration tests injected matching statistics
manually and did not establish end-to-end calibration correctness.

Now `quantize_static` collects selected input/output tensors through real tract
inference and inserts QuantizeLinear/DequantizeLinear pairs on Conv activation
boundaries. Statistics are keyed by original ONNX tensor names, resolved before
tract optimization and associated with ordered output slots. Weight ranges always
come from weight values. There is no string-name matching between weights and
activation statistics.

## Supported contract

The [operator-aware MatMul/Gemm axes](OPERATOR_QUANTIZATION.md) apply to weight-only
quantization. The existing static activation entry points retain the Conv-only
contract below. A separate experimental [matrix activation path](MATRIX_CALIBRATION.md)
now supports fixed FP32 `[1,K]` inputs through Rust and CLI.

- One FP32 model input with fixed positive NCHW dimensions and batch size 1.
- ONNX default-domain opset >= 13. Convert older models before calibration.
- Inline FP32 Conv weights. Weight quantization may be per-tensor or axis-0
  per-channel, symmetric or asymmetric. Other operator weights stay FP32.
- Scalar asymmetric INT8 activation quantization. Zero is included in each range;
  zero-point tensors are INT8, not FP32. `symmetric` controls weights only.
- MinMax, Percentile, Entropy and MSE activation range estimation.
- Nonempty, finite representative FP32 samples matching the model input without
  its batch dimension: dataset shape `[samples, C, H, W]`.

INT4, mixed-bit overrides, multiple/dynamic/non-image inputs, custom domains,
subgraphs, existing QDQ graphs, external tensors and sections the schema cannot
preserve are rejected. tract operator support also limits otherwise eligible
graphs. A different backend or automatic shape guessing is not part of this change.

`excluded_layers` and `min_elements` select Conv weight initializers. Excluded Conv
nodes receive no new local QDQ; they can still consume a quantized activation from
a selected upstream Conv. A selected Conv output is quantized for all consumers.
Shared data inputs reuse one QDQ pair. Conv weights shared with a non-Conv use are
rejected. `layer_bits` is exposed consistently but accepts only 8 in this path.

Graph input/output names are preserved. QDQ nodes appear before their consumers;
generated activation names avoid graph and weight-QDQ names. Transform validation
runs before the existing atomic save. Failures do not replace an existing output.

## Usage

```bash
quantize-rs calibrate model.onnx --data samples.npy -o calibrated.onnx \
    --method minmax --per-channel --symmetric --exclude sensitive.weight
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

```rust
use quantize_rs::{quantize_static, CalibrationDataset, QuantConfig};

fn main() -> Result<(), quantize_rs::QuantizeError> {
    let samples = CalibrationDataset::from_numpy("samples.npy")?;
    let config = QuantConfig::int8().with_per_channel(true).with_symmetric(true);
    quantize_static("model.onnx", "calibrated.onnx", &samples, config)?;
    Ok(())
}
```

## Migration from v0.9

- `Quantizer::with_calibration` is deprecated. Every quantization entry point on
  an object created through it returns `UnsupportedConfig`, including empty or
  nonmatching statistics. Use `Quantizer::new` for weights or `quantize_static`
  for activation QDQ. The old constructor signature remains to provide guidance.
- `ActivationEstimator` statistics use ONNX tensor names instead of optimized node
  names. It remains available for standalone statistics collection.
- CLI `calibrate` no longer generates random data from an unrecognized filename.
- Python `quantize_with_calibration` requires `calibration_data`. Its legacy
  `num_samples` argument remains accepted but is unused with supplied data.
  `sample_shape`, when supplied, must match the dataset. Missing data raises an
  error rather than silently generating noise.
- Python adds `excluded_layers`, `min_elements`, and `layer_bits` at the end of
  the existing argument list, preserving prior positional argument locations.
- Use the ordinary `quantize` path for INT4/native INT4 or mixed precision.
- Dataset constructors/loaders reject empty or zero-sized sample dimensions and
  overflowing element/byte capacities. Random sample generation rejects nonfinite
  bounds or widths. Activation moments now accumulate in f64 with unchanged f32
  getters; histogram-based range estimation is still approximate.

## Validation and practical limits

`tests/static_calibration.rs` runs actual tract inference on generated models.
It checks that substantially different datasets change activation parameters
while weight parameters remain identical, plus filters, shared inputs, naming
collisions, graph connectivity and rejection behavior.

`eval/calibration_smoke_test.py` checks a two-Conv graph in ONNX Runtime with and
without graph optimizations. It uses held-out synthetic inputs from the calibration
distribution and bounds relative output RMSE to 5% for all four range methods.
It also checks CLI validation/benchmark reports and, with `--python`, the actual
installed Python extension's selection controls and errors.

```bash
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs
python eval/calibration_smoke_test.py --binary target/debug/quantize-rs --python
```

These tests demonstrate graph and numerical correctness on small synthetic models.
The additional [labeled CNN evaluation](eval/CNN_EVALUATION.md) reports MNIST
accuracy, latency and process memory on a real pretrained CNN. Neither suite
establishes ImageNet accuracy or universal performance. A production
claim requires representative labeled data, held-out task metrics, and profiling
of the target runtime. Inserting activation QDQ does not guarantee integer kernel
fusion or faster inference.

The implementation follows [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
and [ONNX Runtime static quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#static-quantization).
