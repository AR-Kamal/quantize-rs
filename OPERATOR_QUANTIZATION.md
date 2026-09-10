# Operator-aware weight quantization

The v0.10.0 candidate weight-only `quantize`, `batch`, config-driven CLI and Python
`quantize` paths select direct Conv/MatMul/Gemm weight initializers and infer
their per-channel axis. `Quantizer::quantize_model` uses the same policy.
Existing activation-calibration entry points remain restricted to INT8 Conv
graphs. The separate [experimental matrix activation path](MATRIX_CALIBRATION.md)
now supports fixed FP32 `[1,K]` inputs through Rust and CLI.

## Selection and axes

Eligible tensors are inline, finite FP32 initializers with positive dimensions
and rank >= 2, used directly at input index 1 of a standard-domain operator:

| Operator | Weight layout | Per-channel axis |
|----------|---------------|------------------|
| Conv | `[output_channels, input_channels/group, ...]` | 0 |
| MatMul | `[..., K, N]` as the right operand | Last weight dimension (`rank - 1`) |
| Gemm, `transB=0` or absent | `[K, N]` | 1 |
| Gemm, `transB=1` | `[N, K]` | 0 |

Gemm requires rank-2 weights. `transA`, `alpha` and `beta` remain unchanged;
`transA` does not change the weight axis. Invalid/duplicate `transB` attributes
return an error. MatMul supports higher-rank right operands; one scale per last
dimension aggregates across the reduction and batch dimensions. It does not
produce a separate set of scales per batch.

With per-channel disabled, weights use one scalar scale/zero point. INT8,
INT4 widened storage, native INT4 and per-layer bit overrides retain the selected
axes when per-channel is enabled. Quantized data stays in original row-major
order. Dequantization and reported MSE use the same axis as the ONNX weight DQ node.

The policy follows [ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html),
[Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) and
[DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html).

## Shared uses and boundaries

- A shared weight is quantized once when all consumers use a supported weight
  position with the same axis. Conflicting axes are rejected, even in per-tensor
  mode, to keep a consistent selection policy. No initializer cloning is performed.
- Sharing a selected weight with Gather, Add, a left MatMul operand, a custom
  operator, or another unsupported use returns an error. Selected weights exposed
  as graph outputs are rejected too. The diagnostic identifies the initializer;
  excluding it keeps it FP32 and skips these shared-use checks.
- Exclusions and minimum element counts are applied before shared-use checks.
  Unrelated embeddings, broadcast constants, biases, normalization parameters,
  left-hand-only weights, and unused initializers remain unchanged.
- Weights reached through Reshape/Transpose are skipped. This first implementation
  does not trace constant expressions or infer their transformed layouts.
- Graphs containing subgraphs are rejected when there are selected weights;
  lexical capture/shadowing analysis is not implemented. Selected external tensors,
  malformed payloads and occupied generated names return errors before saving.
- Selection is not full ONNX schema validation, model-wide integer computation,
  or a guarantee of Transformer accuracy or runtime acceleration.

## API compatibility and migration

No fields were added to `QuantConfig`, `WeightTensor`, or `QdqWeightInput`.
Existing tensor constructors and `Quantizer::quantize_weights` retain their
axis-0 behavior because they receive no graph context. `extract_weights` remains
the raw rank-based extractor for inspection and advanced use.

Use the model-aware path for inference of roles and axes:

```rust
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};

fn main() -> Result<(), quantize_rs::QuantizeError> {
    let mut model = OnnxModel::load("model.onnx")?;
    let config = QuantConfig::int8().with_per_channel(true).with_symmetric(true);
    let outputs = Quantizer::new(config).quantize_model(&model)?;
    let qdq = outputs.into_iter().map(|output| output.qdq).collect::<Vec<_>>();
    model.save_quantized(&qdq, "quantized.onnx")?;
    Ok(())
}
```

To inspect selected weights without decoding them twice, call
`model.select_weights(&config)` and then
`Quantizer::new(config).quantize_selected_weights(&selected)`.
The new `SelectedWeight` type has private fields and `weight()`/`axis()` getters.
Keep selections associated with the unchanged source model when saving outputs.

For explicit tensor math use
`QuantizedTensor::from_f32_per_channel_axis(data, shape, axis, symmetric)`
(also available on `QuantizedTensorInt4`). `axis()` reports the stored axis.
Old constructors still use axis 0; existing per-tensor constructors report `None`.

The CLI/Python argument lists do not change, but their selection behavior does:
fewer unrelated tensors may be quantized, and conflicting models now fail rather
than receiving a guessed axis. This is part of the next pre-1.0 minor release's
migration, not a republish of v0.9.0.

## Validation

Rust regressions cover independent per-channel reference calculations for both
bit widths, noncontiguous axes, odd INT4 packing, malformed axes/shapes, role
selection, batching, compatible/conflicting sharing, filters, and layer overrides.

```bash
python -m unittest discover -s eval -p test_operator_axes_smoke.py -v
python eval/operator_axes_smoke_test.py --binary target/debug/quantize-rs
python eval/operator_axes_smoke_test.py --binary target/debug/quantize-rs --python
```

Use an `.exe` path on Windows. The runtime suite exercises non-square MatMul,
batched/broadcast weights, a vector left operand, embeddings, shared weights,
all four Gemm transpose combinations and non-default alpha/beta. It compares
optimized/unoptimized ONNX Runtime outputs with FP32 activations, checks
unselected tensors and public output names, and verifies rejected configurations
preserve existing files.
INT8 asymmetric/symmetric per-channel, INT8 per-tensor, widened INT4 and native
INT4 with a layer override are covered. `--python` requires the newly built wheel.

Strict parity retains `atol=1e-5, rtol=1e-4` and sets the optimized session's
`session.qdq_matmulnbits_accuracy_level` to `1` (FP32 activations), without
disabling graph optimizations. ORT 1.29 can fuse weight-only DQ/MatMul into
MatMulNBits with the default accuracy level `4`, allowing additional INT8
activation quantization. That default execution is tested in a separate session:
all three quantized execution modes must stay below the original relative RMSE
limits against the source FP32 graph (3% for INT8, 35% for INT4), on every sample
and output. See ORT's [session configuration](https://github.com/microsoft/onnxruntime/blob/v1.29.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
and [MatMulNBits accuracy levels](https://github.com/microsoft/onnxruntime/blob/v1.29.0/docs/ContribOperators.md#com.microsoft.MatMulNBits).
These settings apply to the test sessions; exported models retain their normal
ONNX graph and consumers choose their own runtime settings.

These generated graph checks establish local numerical/serialization behavior.
The separate [MNIST report](eval/CNN_EVALUATION.md) remains evidence for the Conv
calibration path. The [GPT-2 evaluation](eval/GPT2_EVALUATION.md) now verifies
direct-weight coverage, bounded held-out perplexity and runtime profiling for one
export. Its matrix kernels remain floating point; broader model validation and
Transformer activation calibration remain future work.
