# Architecture

This document describes the internal structure of quantize-rs for contributors and AI assistants working on the codebase.

## Overview

quantize-rs is a Rust library + CLI + Python module that converts float32 ONNX neural network models to INT8 or INT4 using post-training quantization. It outputs standard QDQ (DequantizeLinear) graphs that load directly in ONNX Runtime.

**Primary target:** computer-vision (CNN) ONNX models -- ResNet, MobileNet, SqueezeNet, and similar architectures. The weight-only path (`Quantizer::quantize_model` over an `OnnxModel`) is model-agnostic and operates on any FP32 initializer. The activation-calibration path uses `tract-onnx` for inference, which has CNN-centric op coverage; transformer / LLM / sequence models can hit unsupported ops or shape-detection mismatches there.

```
                  ┌──────────────────────────────────────────────┐
                  │              User Interfaces                 │
                  │                                              │
                  │  src/main.rs        CLI (clap)               │
                  │  src/python.rs      Python bindings (PyO3)   │
                  └──────────┬──────────────────┬────────────────┘
                             │                  │
                  ┌──────────▼──────────────────▼────────────────┐
                  │              Library (src/lib.rs)             │
                  │                                              │
                  │  quantization/   Core quantization math      │
                  │  calibration/    Calibration pipeline         │
                  │  onnx_utils/     ONNX load/save/validate     │
                  │  config.rs       YAML/TOML config parsing    │
                  └──────────────────────────────────────────────┘
```

## Module map

### `build.rs`

Compiles the vendored `proto/onnx.proto3` schema at build time using `protox` (pure-Rust protoc replacement) and `prost-build`. Outputs `$OUT_DIR/onnx.rs` containing generated Rust types. No system `protoc` binary required.

### `proto/onnx.proto3`

Vendored ONNX protobuf schema matching the official ONNX spec field numbers. Kept in-tree so the build is fully reproducible without network access.

### `src/onnx_proto.rs`

Thin wrapper that `include!`s the prost-generated `$OUT_DIR/onnx.rs`. All ONNX protobuf types (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, etc.) are accessed through this module.

### `src/lib.rs`

Library root. Re-exports all public types: `OnnxModel`, `Quantizer`, `QuantConfig`, `CalibrationDataset`, `ActivationEstimator`, `QdqWeightInput`, etc. Conditionally compiles the `python` module behind the `python` feature flag.

### `src/main.rs`

CLI binary entry point. Defines the clap `Commands` enum (Quantize, Batch, Validate, Info, Benchmark, Config, Calibrate) and dispatches to `cli::commands`.

### `src/cli/`

- **`mod.rs`** -- re-exports `commands`
- **`commands.rs`** (~1390 lines) -- implementation of all CLI commands. Each command is a public function (`quantize`, `batch`, `validate`, `info`, `benchmark`, `run_config`, `calibrate`). This file is the heaviest consumer of the library API. All four quantize entry points route through `Quantizer::quantize_model`. `batch` builds a rayon `ThreadPoolBuilder` for `--jobs N` and serializes stdout through a `Mutex` so concurrent progress lines do not interleave (the lock guard recovers from `PoisonError` via `into_inner` so a panicking worker doesn't cascade-kill its siblings). `validate` / `info` / `benchmark` have human-and-JSON output paths sharing a single computation, with the JSON path emitting `ValidateReport` / `InfoReport` / `BenchmarkReport` serde structs. `benchmark`'s structure-preservation check is QDQ-aware (accounts for inserted `DequantizeLinear` nodes). `batch`'s terminal banner distinguishes "all skipped" from "all succeeded", and `batch` returns `Err` (non-zero CLI exit) when at least one model fails, when zero files match the glob, or when a model loads successfully but contains no FP32 weights. `run_config` propagates per-model failures with the same `Err` discipline and honours `BatchConfig.jobs` for the parallel path. `calibrate` reloads the saved file and runs `validate_connectivity` after saving, matching the `quantize` UX.

### `src/quantization/`

- **`mod.rs`** (~880 lines) -- the core quantization engine and most of the unit tests (40+)

Key types:
- `QuantConfig` -- configuration (bits, per_channel, **symmetric**, calibration_method, excluded_layers, layer_bits, min_elements). Helpers: `should_quantize(name, n)`, `bits_for_layer(name)`. Public fields with `Default` impl; downstream callers construct via `QuantConfig { /* fields */, ..Default::default() }`.
- `QuantParams` / `QuantParamsInt4` -- scale + zero_point for INT8 / INT4. Compute from a float range, quantize single values, dequantize back. Symmetric variants: `from_range_symmetric`, `from_f32_symmetric`, `from_f32_per_channel_symmetric`, `from_f32_with_range_symmetric`.
- `QuantizedTensor` / `QuantizedTensorInt4` -- quantized tensor with data, shape, params, and optional per-channel params. Constructors: `from_f32`, `from_f32_with_range`, `from_f32_per_channel`, plus the `_symmetric` variants above.
- `QuantizedTensorType` -- `#[non_exhaustive]` enum over Int8/Int4, unifying the two tensor types with shared methods (`data()`, `bits()`, `get_all_scales_zero_points()`, `is_per_channel()`, `data_ref()`). The `non_exhaustive` marker means downstream `match` arms must include a `_ => ...` catch-all.
- `Quantizer` -- high-level orchestrator. Holds config + optional calibration stats. Methods: `quantize_tensor` (weight-only), `quantize_tensor_with_name` (calibration-aware), and **`quantize_model(&self, model: &OnnxModel) -> Result<Vec<QuantizedWeightOutput>>`** -- the consolidated entry point that runs the full filter / parallel / `layer_bits` pipeline and is now the single source of truth for the four CLI commands and both Python functions.
- `QuantizedWeightOutput { qdq: QdqWeightInput, quantized_size_bytes: usize, mse: f32 }` -- `#[non_exhaustive]` result type from `quantize_model`.

Standalone functions:
- `pack_int4` / `unpack_int4` -- bit-pack two INT4 values per byte (internal layout: `val1` in high nibble)

### `src/calibration/`

- **`mod.rs`** -- `CalibrationDataset` (load from .npy via ndarray-npy, generate random, or construct from vectors). Behind feature `safetensors-input`: `from_safetensors`, `from_safetensors_named` (load tensors directly from HuggingFace `.safetensors` files). Validates inputs (empty shape, 1-D rejection, shape mismatches).
- **`stats.rs`** -- `ActivationStats` (min, max, mean, std, histogram). Fixed 256-bin histogram with rebinning on range expansion. Incremental updates via Chan's parallel algorithm for std; the merged mean is computed entirely in `f64` and cast to `f32` once at the end. `calculate_optimal_range` dispatches to MinMax/Percentile/Entropy/MSE methods. **`calculate_optimal_range_from_stats`** is the histogram-direct variant -- KL/MSE search reads the histogram directly instead of regenerating samples through an RNG, which is both deterministic and ~2x faster. `histogram_kl_divergence`, `histogram_quantization_mse`, `optimize_kl_from_stats`, `optimize_mse_from_stats`.
- **`methods.rs`** -- `#[non_exhaustive]` `CalibrationMethod` enum (MinMax, Percentile(f32), Entropy, MSE). `Display` renders the lowercase keyword (`"minmax"`, `"entropy"`, `"mse"`) and `Percentile(p)` as `"percentile:p"`. `FromStr` accepts the same forms plus the back-compat bare `"percentile"` (defaults to 99.9). Percentiles outside `[0, 100]` are rejected at parse time. `Default` is `MinMax`.
- **`inference.rs`** -- `ActivationEstimator`. Loads the model with tract, exposes all intermediate layer outputs, runs calibration samples through inference, and collects `ActivationStats` per layer. Public methods: `new` / `from_path`, `calibrate` (prints progress), `calibrate_quiet` (silent — for embedded library use), `get_layer_stats` (borrowed), `into_layer_stats` (owned), `into_model`, `model`. The output name mapping is built after tract optimization to avoid desync from node fusion. tract's op coverage is CNN-centric; transformer / LLM models may fail to load here.

### `src/onnx_utils/`

- **`mod.rs`** -- `OnnxModel` (load, **load_mmap** [feature `mmap`], **from_bytes**, info, extract_weights, **count_non_fp32_weight_initializers**, save_quantized, **save_quantized_with_options**, validate_connectivity, load_quantized_info, input_shapes, total_size_bytes). File size guard (10 GB). `save_quantized_with_options` writes atomically: encode to a unique pid-tagged sibling `*.quantize-rs.tmp`, `sync_all()`, then `rename` into place — a crash mid-write leaves no torn file at the target path, and a failed write or rename removes the temp file. Repeated saves on the same instance dedup `quantize_rs.bits.*` metadata entries before re-emitting. `WeightTensor` and `ModelInfo` (`#[non_exhaustive]`; now carries `opset_version`) structs. `extract_weights` skips QDQ scale scaffolding (`{base}_scale` with a `{base}_quantized` sibling) so re-quantizing a quantized model is a clean no-op. `QuantizedWeightInfo` (`#[non_exhaustive]`) for QDQ introspection stores `scales: Vec<f32>` / `zero_points: Vec<i8>`; the convenience accessors `scale()` and `zero_point()` now return `Option<f32>` / `Option<i8>` (return `None` on malformed empty input rather than panic). `storage_bytes: usize` populated from `raw_data.len()` for correct native-INT4 byte counting.
- **`graph_builder.rs`** -- QDQ graph transform (`apply_qdq_transform`, **`apply_qdq_transform_with_options`**), graph connectivity validation (`validate_graph_connectivity`), opset version management (`ensure_opset_version` + `upgrade_deprecated_ops`). The QDQ transform runs a pre-flight validation pass: if any requested weight is missing from the initializer list it returns `Err` *before* mutating the graph, with a special diagnostic when the weight is already present as `{name}_quantized` ("already quantized — load a fresh OnnxModel"). When bumping opset, automatically handles: `BatchNormalization.spatial` removal (opset 9), `Dropout.ratio` attribute->input migration (opset 12 — synthetic initializer name includes the node index to avoid collisions), `Softmax` default axis change (opset 13). `QdqWeightInput` struct (the interface between quantization output and ONNX save) derives `Clone`. **`SaveOptions`** is `#[non_exhaustive]` with a `with_native_int4(bool)` builder controlling whether INT4 weights are widened to INT8 bytes (default) or written as ONNX `DataType::Int4` in opset 21 (native). `ConnectivityReport` with `summary()` for CLI output. 17+ unit tests with helper graphs. These graph-level functions are `pub(crate)` (they take/return raw `prost` ONNX types); public callers go through the `OnnxModel` save/validate methods.
- **`quantization_nodes.rs`** (crate-internal — `pub(crate)`) -- low-level ONNX protobuf builders: `build_dequantize_linear_node`, `build_quantized_weight_tensor`, `build_scale_tensor`, `build_zero_point_tensor`. The weight and zero-point builders take a **`StorageFormat`** parameter (`#[non_exhaustive]` enum with `Int8Widened` and `NativeInt4` variants); `pack_int4_onnx` / `unpack_int4_onnx` implement the ONNX wire layout (even index in low nibble), distinct from the internal `pack_int4` layout. `DequantLinearNames` generates canonical naming.

### `src/config.rs`

YAML/TOML configuration file support. `Config`, `ModelConfig`, `BatchConfig` structs with serde. `BatchConfig` carries a `jobs: usize` field (defaults to 1) honoured by `run_config` for parallel batch processing. Validation logic for bits, paths, and batch settings.

### `src/python.rs`

PyO3 bindings, built with `pyo3/abi3-py39` so one wheel covers Python 3.9+. Three functions exposed: `quantize()`, `quantize_with_calibration()`, `model_info()`. Wraps the library API with Python-friendly error handling (PyRuntimeError). Both quantization functions wrap their heavy I/O and compute in `py.allow_threads(...)` so other Python threads can run concurrently; both route through `Quantizer::quantize_model`, sharing the filter / parallel / `layer_bits` pipeline with the CLI. The calibration path calls `ActivationEstimator::calibrate_quiet` to suppress library-internal `println!` output and accepts `native_int4` and `symmetric` kwargs.

### `src/errors.rs`

`QuantizeError` is a `#[non_exhaustive]` enum with 7 variants (`InvalidTensor`, `UnsupportedConfig`, `ModelLoad`, `ModelSave`, `GraphTransform`, `Calibration`, `Config`). Exported as `pub mod errors` and `pub use errors::QuantizeError` from the crate root. All public library functions return `crate::errors::Result<T>`. Manual `Display` + `Error` impls (no `thiserror` dependency). The CLI and Python bindings auto-convert via `?` (to `anyhow::Error`) and `.map_err()` (to `PyRuntimeError`) respectively. The `non_exhaustive` marker means downstream `match` arms must include a `_ => ...` catch-all.

## Data flow

### Weight-only quantization

```
OnnxModel::load(path)                                 (or load_mmap on `mmap` feature)
    → Quantizer::new(config)
    → quantizer.quantize_model(&model)                → Vec<QuantizedWeightOutput>
        → internal: extract_weights → filter (excluded_layers, min_elements)
        → rayon par_iter over kept tensors
        → per-tensor: quantize_tensor[_with_name]   → QuantizedTensorType
        → bundle scales/zero_points into QdqWeightInput
    → collect into Vec<QdqWeightInput>
    → model.save_quantized(data, path)                (or save_quantized_with_options)
        → apply_qdq_transform(graph, data, opts)
            → remove FP32 initializers
            → add INT8 (or native-INT4 if opts.native_int4) initializers + DequantizeLinear nodes
        → write protobuf to disk
```

The CLI commands (`quantize`, `batch`, `calibrate`) and both Python functions all go through `quantize_model`. Direct `quantize_tensor` / `quantize_tensor_with_name` calls remain available for callers that need to drive the loop themselves (custom filtering, partial models, etc.).

### Calibration-based quantization

```
OnnxModel::load(path)
    → ActivationEstimator::new(model, path)
        → tract loads ONNX, exposes all layer outputs, optimizes
    → estimator.calibrate(dataset)
        → for each sample: run inference, capture activations, update stats
    → stats = estimator.into_layer_stats()
    → model = reload or recover via into_model()
    → Quantizer::with_calibration(config, stats)
    → quantizer.quantize_tensor_with_name(name, data, shape)
        → looks up activation stats for this layer
        → calculate_optimal_range(data, method)
        → quantize with calibrated range
    → model.save_quantized(...)
```

### QDQ graph pattern

For each quantized weight `conv1.weight`:

```
Before:
    [conv1.weight (FP32)] ──→ Conv node

After:
    [conv1.weight_quantized (INT8)]  ─┐
    [conv1.weight_scale (FP32)]      ─┤
    [conv1.weight_zp (INT8)]         ─┤
                                      └─→ DequantizeLinear ──→ [conv1.weight] ──→ Conv node
```

The DequantizeLinear output name matches the original, so all downstream nodes are unchanged.

## Key dependencies

| Crate | Purpose |
|-------|---------|
| `prost` | ONNX protobuf decode/encode at runtime |
| `protox` + `prost-build` | Compile `proto/onnx.proto3` at build time (no system `protoc`) |
| `tract-onnx` + `tract-core` (feature `calibration`) | ONNX inference engine for activation calibration |
| `ndarray` + `ndarray-npy` (feature `calibration`) | Calibration data loading from .npy files |
| `memmap2` (feature `mmap`) | Memory-mapped ONNX loading for multi-GB models |
| `safetensors` (feature `safetensors-input`) | Load calibration tensors from HuggingFace `.safetensors` |
| `clap` | CLI argument parsing |
| `pyo3` (feature `python`) | Python bindings |
| `rayon` | Parallel weight quantization + parallel batch processing |
| `anyhow` | Error handling (CLI binary only; library uses `QuantizeError`) |
| `serde` + `serde_yaml` + `serde_json` + `toml` | Config file parsing + `--format json` CLI output |
| `colored` | CLI colored output |
| `proptest` (dev) | Property-based tests in `tests/property_tests.rs` |
| `criterion` (dev) | Benchmarks in `benches/quantization.rs` |
| `libfuzzer-sys` (fuzz) | Fuzz harness for `OnnxModel::from_bytes` (`fuzz/`, separate workspace) |

## Test structure

**169 tests passing on default features** (109 unit + 4 CLI-binary + 39 integration + 17 property-based). With `--all-features`, total rises to **172** (adds 1 mmap integration test + 2 safetensors lib tests). 5 tests marked `#[ignore]` require ONNX model files on disk or specific environments.

### Unit tests (109 lib + 4 CLI-binary)

- `src/quantization/mod.rs` -- ~45 tests covering INT8/INT4 quantization, per-channel, packing, compression, error bounds, symmetric round-trip, `should_quantize`, `bits_for_layer`
- `src/onnx_utils/graph_builder.rs` -- ~17 tests covering QDQ transform, connectivity validation, opset management, deprecated attribute upgrade, `SaveOptions::native_int4`
- `src/onnx_utils/quantization_nodes.rs` -- ~12 tests covering protobuf node/tensor builders for both `Int8Widened` and `NativeInt4` storage formats, plus `pack_int4_onnx` / `unpack_int4_onnx` round-trip
- `src/calibration/stats.rs` -- activation stats, percentile, histogram-direct KL/MSE
- `src/calibration/mod.rs` -- dataset construction (npy + safetensors)
- `src/calibration/methods.rs` -- 4 tests covering `Display`/`FromStr` round-trip, `"percentile:NN"` parsing, invalid-percentile rejection, and keyword discriminant identity
- `src/config.rs` -- YAML/TOML parsing, validation (including empty layer_bits key rejection)
- `src/lib.rs` -- version string
- `src/main.rs` -- 4 CLI-binary unit tests: calibration-method parsing (incl. `percentile:NN`) and `--layer-bits` conflict handling

### Integration tests (`tests/integration.rs`, 40)

End-to-end tests that construct ONNX models in memory (no model files on disk):

- `test_quantize_simple_model_int8` -- full pipeline: build model, quantize INT8, save, reload, validate connectivity + QDQ info
- `test_quantize_simple_model_int4` -- same pipeline with INT4
- `test_quantize_per_channel` -- two-weight model, per-channel quantization, verify per-channel scale shapes
- `test_round_trip_quantization_accuracy` -- known data, quantize/dequantize, verify MSE within tolerance
- `test_error_variants_are_correct` -- invalid inputs produce correct `QuantizeError` variants
- `test_config_round_trip` -- YAML/TOML parsing + validation + error cases
- `test_config_layer_bits` -- per-layer bit-width overrides
- `test_mixed_precision_quantization` -- INT4 + INT8 mix in same model
- `test_multilayer_min_elements` -- tensors below `min_elements` threshold are left in FP32
- `test_multilayer_excluded_layers` -- explicitly excluded layers are skipped
- `test_multilayer_full_round_trip` -- multi-layer model: quantize, save, reload, validate QDQ structure
- `test_multilayer_compression_ratio` -- verify file size reduction on a multi-weight model
- `test_dual_input_initializer_model` -- model with weights listed as both initializers and graph inputs
- `test_calibrated_quantization_uses_stats` -- calibrated scale < uncalibrated scale
- `test_calibrated_quantization_with_method` -- MinMax calibration method
- `test_calibrated_quantization_fallback_no_stats` -- fallback when layer has no stats
- `test_calibrated_quantization_int4` -- calibrated INT4 end-to-end
- `test_calibrated_full_pipeline` -- full pipeline: calibrate + quantize + save + reload + validate connectivity
- Native INT4 tests: `data_type == Int4`, opset bumps to 21, packed `raw_data` matches the ONNX wire layout, per-channel zero-points round-trip
- Symmetric tests: zero_points all 0 after save/reload
- mmap test (feature `mmap`): `OnnxModel::load_mmap` produces an output identical to `OnnxModel::load`

### Property-based tests (`tests/property_tests.rs`, 17)

Uses `proptest` to exercise numeric edge cases:

- `quantize(dequantize(v)) ≈ v` for exact grid points (INT8 and INT4)
- `|dequantize(quantize(v)) - v| ≤ scale/2` for all values in range
- `pack(unpack(data)) == data` for all valid INT4 data
- Quantization never panics on any finite f32 input
- Per-channel quantization error bounds

### Benchmarks (`benches/quantization.rs`)

Criterion benchmark groups:

- `quantize_throughput` -- INT8/INT4 at 1K/100K/1M elements with `Throughput::Elements`
- `per_channel_vs_per_tensor` -- comparison across 4 variants on a [64, 27] tensor
- `pack_int4` -- raw pack/unpack throughput at 10K/100K/1M elements
- `quantize_model` -- full Quantizer loop over 8 synthetic weight tensors

Run with `cargo bench`.

## File size reference

Approximate line counts (subject to drift; rerun `wc -l` for exact figures).

| File | Lines | Role |
|------|-------|------|
| `src/quantization/mod.rs` | ~2170 | Core math + symmetric variants + `quantize_model` + ~45 tests |
| `tests/integration.rs` | ~1750 | 40 end-to-end integration tests |
| `src/cli/commands.rs` | ~1390 | All CLI command implementations + `--jobs` parallel batch + JSON output paths + non-zero exit propagation |
| `src/onnx_utils/graph_builder.rs` | ~990 | QDQ transform + `SaveOptions` + opset upgrade + pre-flight validation + tests |
| `src/calibration/stats.rs` | ~700 | Histogram, stats, calibration methods, histogram-direct KL/MSE |
| `src/onnx_utils/mod.rs` | ~700 | Model load/save/validate + atomic save + `load_mmap` + `from_bytes` + `save_quantized_with_options` |
| `src/onnx_utils/quantization_nodes.rs` | ~465 | Protobuf builders + `StorageFormat` + ONNX-wire INT4 packing + tests |
| `src/calibration/inference.rs` | ~420 | Activation estimator (tract) + `calibrate_quiet` |
| `src/calibration/mod.rs` | ~400 | CalibrationDataset (npy + optional safetensors) |
| `src/main.rs` | ~380 | CLI entry + clap definitions (incl. `--jobs`, `--format`, `--native-int4`, `--symmetric`) |
| `tests/property_tests.rs` | ~350 | 17 proptest property-based tests |
| `src/config.rs` | ~350 | Config parsing (incl. `BatchConfig.jobs`, native_int4, symmetric, excluded_layers, layer_bits, min_elements) |
| `src/python.rs` | ~300 | Python bindings (abi3-py39, GIL released around heavy work) |
| `benches/quantization.rs` | ~230 | Criterion benchmarks (4 groups) |
| `src/errors.rs` | ~100 | `#[non_exhaustive]` `QuantizeError` enum (7 variants, exported) |
| `src/calibration/methods.rs` | ~125 | `#[non_exhaustive]` `CalibrationMethod` enum + `Display`/`FromStr` round-trip + tests |
| `eval/benchmark_gpt2.py` | ~440 | GPT-2 small end-to-end (export + weight-only quantize + perplexity) |
| `eval/validate_models.py` | ~395 | Real-world model validation (ResNet-18, MobileNetV2, SqueezeNet) |
| `proto/onnx.proto3` | ~180 | Vendored ONNX schema |
| `fuzz/fuzz_targets/onnx_load.rs` | ~10 | `cargo-fuzz` target for `OnnxModel::from_bytes` |
| `build.rs` | ~20 | Proto compilation (protox + prost-build) |
| `src/onnx_proto.rs` | ~5 | Includes prost-generated ONNX types |

## Conventions

- **Error handling:** `QuantizeError` enum at all public library boundaries (`#[non_exhaustive]`). CLI uses `anyhow::Result` (auto-converts via `?`). Python uses `.map_err()` to `PyRuntimeError`. The CLI's `quantize` / `batch` / `config` subcommands return non-zero exit codes on missing input, empty FP32 weight set, glob with no matches, or any per-model failure; `validate` returns non-zero when validation fails.
- **API stability for v1.0:** `#[non_exhaustive]` is applied to public enums (`QuantizeError`, `QuantizedTensorType`, `CalibrationMethod`) and result-only structs (`SaveOptions`, `QuantizedWeightInfo`, `QuantizedWeightOutput`, `ModelInfo`). Frequently-constructed structs (`QuantConfig`, `QdqWeightInput`) remain exhaustive with public fields plus a `Default` impl; new fields are added on a minor bump. The `QuantRange` marker trait is **sealed** (cannot be implemented downstream). The low-level, `prost`-coupled helpers (`apply_qdq_transform*`, `validate_graph_connectivity`, `ensure_opset_version`, and the whole `quantization_nodes` module incl. `StorageFormat`) are `pub(crate)`, so no `prost`/`onnx_proto` type appears in the public API.
- **Field visibility:** Public structs have private fields with getter methods. Internal access uses `pub(crate)`.
- **NaN safety:** All min/max folds filter non-finite values. Quantize functions map NaN/Inf to zero_point. `from_range()` guards against NaN before `as i8` cast. `ActivationStats::update` does the merged mean entirely in `f64` and casts to `f32` once at the end.
- **Axis convention:** Per-channel quantization always uses axis 0 (output channel dimension).
- **ONNX storage:** INT4 values are widened to INT8 bytes by default (compatible with opset 10+). `SaveOptions::with_native_int4(true)` writes them as `DataType::Int4` and bumps the opset to 21.
- **Naming:** QDQ initializers follow `{name}_quantized`, `{name}_scale`, `{name}_zp` convention.
- **Opset management:** `save_quantized()` bumps to minimum needed opset (10 per-tensor, 13 per-channel, 21 for native INT4). `upgrade_deprecated_ops()` handles breaking attribute changes across opset boundaries automatically (`BatchNormalization.spatial` @9, `Dropout.ratio` attribute→input @12, `Softmax` default axis @13).
- **Atomic writes:** `save_quantized_with_options` writes to a unique pid-tagged sibling `*.quantize-rs.tmp`, calls `sync_all()`, then `rename`s into place. Crash-safe — readers see either the old file or the fully-written new file; a failed write or rename removes the temp file.
