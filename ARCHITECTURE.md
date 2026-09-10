# Architecture

quantize-rs is a Rust library, CLI and Python extension for ONNX quantization.
The ordinary path compresses FP32 weights to INT8/INT4. The v0.10.0 static
calibration path additionally quantizes selected Conv activation boundaries.
See [CALIBRATION.md](CALIBRATION.md) for its precise restrictions and migration.

## Module map

| Module | Responsibility |
|--------|----------------|
| `src/quantization/mod.rs` | Shared INT8/INT4 math, explicit per-channel axes, packing, filtering and parallel weight quantization |
| `src/onnx_utils/selection.rs` | Direct operator weight roles, inferred axes and shared-use validation |
| `src/onnx_utils/mod.rs` | ONNX loading, metadata, initializer extraction, QDQ introspection and atomic saving |
| `src/onnx_utils/graph_builder.rs` | Weight QDQ transform, connectivity validation and opset migration |
| `src/onnx_utils/calibrated.rs` | Static Conv eligibility, activation QDQ insertion, collision-free activation names and transactional candidate transform |
| `src/onnx_utils/quantization_nodes.rs` | Internal protobuf builders and native INT4 wire packing |
| `src/calibration/static_quantization.rs` | Public `quantize_static` orchestration and representative dataset validation |
| `src/calibration/inference.rs` | tract inference, original tensor labels and activation statistics collection |
| `src/calibration/stats.rs` | Histograms and MinMax/Percentile/Entropy/MSE range optimization |
| `src/calibration/mod.rs` | FP32 sample datasets, `.npy` and optional safetensors input |
| `src/config.rs` | YAML/TOML weight-only configurations |
| `src/cli/commands.rs` | CLI workflows and human/JSON reporting |
| `src/python.rs` | PyO3 API; GIL released around heavy work |
| `src/errors.rs` | Typed errors at public library boundaries |
| `proto/onnx.proto3`, `build.rs` | Vendored schema compiled by protox/prost-build without system protoc |

## Weight-only flow

```text
OnnxModel::load
  -> select_weights(config): direct inline FP32 Conv/MatMul/Gemm weights
  -> validate uses and infer axes into private-field SelectedWeight records
  -> Quantizer::new(config).quantize_selected_weights
  -> filter and parallel quantization using each weight's own values
  -> QdqWeightInput blocks
  -> save_quantized_with_options
  -> weight DequantizeLinear nodes and atomic file replacement
```

INT4 is packed internally. It is widened to INT8 bytes by default when saving;
native INT4 requires opset 21. Model-aware per-channel axes are Conv 0, MatMul's
last weight dimension and Gemm 1 (0 for transB=1). The axis-0 path retains its
contiguous loop; other axes use strides without transposing stored values.
`to_f32` uses the same stored axis, including packed INT4 and MSE reporting.
Raw `extract_weights` and `quantize_weights` remain available with their previous
rank-based/axis-0 behavior. See [the operator policy](OPERATOR_QUANTIZATION.md).

## Static activation flow

```text
quantize_static(input_path, output_path, dataset, config)
  -> validate fixed single FP32 NCHW input and representative finite samples
  -> calibration_plan: select supported Conv weights and their tensor edges
  -> Quantizer::new(config): quantize weights from their own ranges
  -> ActivationEstimator::for_tensors: expose only selected activation edges
  -> real tract inference and tensor-name-keyed activation statistics
  -> save_calibrated: build candidate model with activation QDQ
  -> validate connectivity, insert weight DQ, reuse atomic writer
```

tract does not label source outlets automatically, so the estimator explicitly
labels them using original input names. Computational tensor labels are resolved
before optimization; output-slot order preserves the correspondence after
optimization. Node names and initializer names are never used to match activation
statistics to weights. Standalone `ActivationEstimator` also reports tensor names.

Each selected Conv input receives scalar INT8 QDQ, shared by selected consumers
of that edge. A selected upstream Conv output already has QDQ and is reused.
Conv outputs are renamed internally and restored by DQ, preserving original
public output names and downstream references. Q/DQ nodes are inserted in
execution order. Generated activation names avoid existing graph names and
planned weight-QDQ names. Activation zero-point initializers have INT8 dtype;
scales have FP32 dtype, and ranges include zero.

The `symmetric` and `per_channel` flags affect weights only. Activations use
per-tensor asymmetric INT8. Other operators are left floating point. Excluded
layers can consume a quantized output from a selected upstream Conv; exclusion
means no local quantization is inserted for that Conv.

The old `Quantizer::with_calibration` constructor is deprecated. Objects built
through it return `UnsupportedConfig` on quantization, preventing both silent
fallback and accidental use of activation ranges for weights. Its signature is
retained to provide an actionable migration error.

## Boundaries and compatibility

Dataset dimensions and byte capacities are checked before sample splitting or
generation. Activation sums, running means and second moments stay in f64;
the public mean/std getters still return f32. Histograms retain their existing
approximate range-estimation behavior.

`try_load_quantized_info` validates dimensions and scale/zero-point payload
lengths and returns typed errors. CLI validation/benchmark reports use this path.
The older `load_quantized_info` signature remains; malformed metadata causes a
warning and an empty list. Missing initializers retain scalar defaults; malformed
present initializers never allocate fallback vectors derived from their shapes.

`calibration` is enabled by default and includes tract/ndarray. Without it,
weight-only quantization, range math and datasets built from vectors remain
available. Optional features include `python`, `mmap`, and `safetensors-input`.
The `python` feature alone does not expose activation calibration functions.

`QuantConfig` and `QdqWeightInput` have public fields and implement `Default`.
Their field sets are frozen for v1.x: adding fields can break downstream exhaustive
literals. Future options must use separate types/new methods or a major release.
New extensible option types should use constructors/builders with private fields
or `#[non_exhaustive]` from introduction. `Default` alone does not grant compatible
field additions. Public result-only types and enums use `#[non_exhaustive]` where
already declared; `QuantRange` is sealed. Raw `onnx_proto` is doc-hidden and
explicitly outside the stable API contract.

Atomic saving uses a unique sibling temporary file, syncs it, then renames it.
The static path transforms a candidate model and publishes it only after checks.
Invalid configurations/data do not replace existing output files. The candidate
clone has a memory cost; this path does not claim zero-copy loading or large-model
memory optimization.

Unsupported external data and unpreserved sections are errors in static
calibration. The older weight-only save path retains its warning-based behavior.
The static path requires opset >= 13 rather than attempting broad version migration.

## Validation

The experimental `quantize_static_matrix` path has separate graph/input planning
in `onnx_utils/matrix_calibrated.rs`, with a fixed FP32 `[1,K]` contract. It reuses
tract tensor statistics and the common activation-QDQ writer, selecting matrix
operators explicitly. The Conv entry point still selects Conv only. See
[MATRIX_CALIBRATION.md](MATRIX_CALIBRATION.md).

- Rust unit/property tests cover weight math, range optimization and ONNX utilities.
- `tests/integration.rs` covers weight saving, errors and rejection of legacy
  calibrated-weight calls.
- `tests/static_calibration.rs` runs actual tract inference on generated graphs;
  it checks dataset sensitivity, weight independence, per-channel settings,
  filtering, shared edges, naming collisions and failure preservation.
- `tests/operator_selection.rs` covers weight roles, inferred axes, sharing,
  filters and mixed-bit propagation; `eval/operator_axes_smoke_test.py` checks
  MatMul/Gemm serialization, ONNX Runtime optimization parity and Python routing.
- `tests/matrix_calibration.rs` checks the separate matrix activation API through
  real inference. `eval/matrix_calibration_smoke_test.py` verifies held-out output
  error, runtime parity, integer kernels and documented fan-out fusion limits.
- `eval/ort_smoke_test.py` executes four weight-only storage configurations in ORT.
- `eval/calibration_smoke_test.py` checks static Conv output accuracy on held-out
  synthetic samples with ORT optimizations enabled and disabled, all four range
  methods, CLI reports and Python API parity.
- `eval/cnn_evaluation.py` compares all three variants on labeled image data.
  The MNIST preset uses 1,000 training examples and all 10,000 held-out test images.
  See [the evaluation report](eval/CNN_EVALUATION.md) for results and limitations.
- `eval/benchmark_gpt2.py` delegates to `gpt2_evaluation.py` for a pinned WikiText-2
  INT8 evaluation. `weight_coverage.py` audits candidates and verifies them against
  actual Rust QDQ output, including unchanged initializers. Separate processes
  measure each variant; ORT profiling records executed matrix input types.
  `test_gpt2_evaluation.py` checks token accounting, loss math, gates and coverage.
  See [the bounded GPT-2 report](eval/GPT2_EVALUATION.md).
- `eval/validate_models.py` retains historical weight-only model evaluations.
  Weight-only results do not establish static activation calibration accuracy.

The synthetic checks are correctness regressions, not task-accuracy or speed
claims. Real-dataset accuracy and runtime profiling are still required before
promoting the activation path to a production stability promise.
