# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-15

### Added

- Per-channel DequantizeLinear serialization: `save_quantized` now writes 1-D scale/zero_point tensors and the `axis` attribute so per-channel quantization parameters survive to the ONNX file
- `QdqWeightInput` named struct replaces the raw 5-element tuple in `save_quantized`, making the API self-documenting
- `QuantizedTensorType::get_all_scales_zero_points()` returns per-channel scales and zero-points (single-element vectors for per-tensor)
- `QuantizedTensorType::is_per_channel()` and `data_ref()` (borrow without cloning) methods
- Parallel weight quantization via rayon in CLI `quantize` and `calibrate` commands
- `validate` command detects QDQ models and adjusts node/input count expectations accordingly
- `validate` command uses `load_quantized_info()` for QDQ weight validation (scale sanity checks) instead of comparing raw initializer bytes
- `OnnxModel::input_shapes()` method extracts input tensor shapes from the ONNX protobuf for reliable auto-detection
- NaN/Inf guards throughout the quantization and calibration pipeline: `QuantParams::quantize()`, min/max folds, `build_histogram`, histogram sorting
- Shape-vs-data length validation in all `from_f32*` tensor constructors
- `extract_channel` returns `Result` with bounds validation (empty shape, zero channels, OOB index, uneven data)
- `CalibrationDataset::random()` validates inputs and returns `Result` (rejects empty shapes, zero samples, invalid ranges)
- `CalibrationDataset::from_numpy()` rejects 1-dimensional arrays with a clear error
- Division-by-zero guards for empty `channel_params` in `to_f32()` dequantization
- File size guard (10 GB limit) in `OnnxModel::load()` to prevent OOM on accidental misuse
- `#[must_use]` annotations on `ConnectivityReport`, `Config::validate()`, `quantization_error()`, `data()`, `bits()`
- `Debug` trait implemented for all public types: `OnnxModel`, `Quantizer`, `CalibrationDataset`, `ActivationEstimator`, `QdqWeightInput`, `ConnectivityReport`
- `Display` and `FromStr` implementations for `CalibrationMethod`, eliminating duplicated parsing in CLI, Python, and main
- Getter methods for private fields on `QuantParams`, `QuantizedTensor`, `ActivationStats`

### Fixed

- Version string in CLI banner was hardcoded to `v0.1.0` instead of reading from `Cargo.toml`
- `validate` command now runs graph connectivity validation (the check was built in v0.3.0 but never wired into the CLI)
- `ActivationStats::update()` now correctly tracks standard deviation across incremental updates using Chan's parallel algorithm; previously `std` was frozen at its initial value, corrupting calibration sampling
- Unsigned subtraction panic in `validate` and `benchmark` commands when the quantized model is larger than the original (possible with QDQ overhead on small models)
- Batch command output filenames now use the actual bit width (`_int4` / `_int8`) instead of always `_int8`
- Python `quantize_with_calibration()` no longer loads the model twice when auto-detecting input shape; uses `into_model()` instead of reloading from disk
- Removed stale TODO comment in `methods.rs` referencing functions that already exist in `stats.rs`
- Per-channel dequantization index-out-of-bounds panic when tensor length is not evenly divisible by channel count
- `sample_from_activation_stats()` now samples from the observed histogram distribution; previously generated uniform samples, corrupting entropy/MSE calibration
- KL divergence calculation now uses dense aligned bins; previously zipped sparse histograms by position, pairing unrelated bins
- `extract_channel()` no longer accepts a `channel_axis` parameter it cannot honor; only axis-0 extraction is supported and the API now reflects that
- `validate` command no longer reports false "VALIDATION FAILED" on correctly quantized QDQ models (node count and input count changes are expected)
- `benchmark` command now uses `load_quantized_info()` for QDQ models instead of extracting weights incorrectly
- Histogram rewritten to use fixed 256 bins with rebinning on range expansion; previous approach caused unbounded bin fragmentation over incremental updates
- `ActivationStats::default()` now uses `f32::INFINITY` / `f32::NEG_INFINITY` sentinels; previously `min=0.0, max=0.0` biased the first incremental update
- `validate --detailed` now detects the actual bit width from QDQ info instead of always re-quantizing as INT8
- `calibrate` CLI command auto-detects model input shape instead of hardcoding `[1, 28, 28]`
- Python shape auto-detection now uses `OnnxModel::input_shapes()` to read shapes from the protobuf instead of parsing string names
- `ActivationEstimator` builds output name mapping after tract optimization, preventing desync when nodes are fused or reordered
- Division-by-zero in `quantize` command compression ratio when quantized size is 0
- `file_name().unwrap()` panic in batch command on root or `..` paths
- Batch status filtering no longer depends on ANSI color escape codes
- Negative i64 ONNX dimensions now clamped to 0 instead of wrapping to large `usize` values
- QDQ save validates `quantized_values` length matches the tensor shape before writing to disk
- `activation_calibration` example no longer reloads model from disk; uses `into_model()` to recover it

### Changed

- **BREAKING:** `save_quantized` signature changed from `&[(String, Vec<i8>, f32, i8, u8)]` to `&[QdqWeightInput]`
- **BREAKING:** `CalibrationDataset::random()` now returns `Result<Self>` instead of `Self`
- **BREAKING:** Struct fields on `QuantParams`, `QuantParamsInt4`, `QuantizedTensor`, `QuantizedTensorInt4`, and `ActivationStats` are now private; use accessor methods instead
- **BREAKING:** `CalibrationMethod::name()` removed; use `Display` formatting (`format!("{}", method)`) instead
- **BREAKING:** `QuantMode` enum removed from public API (was unused)
- **BREAKING:** `cdylib` removed from default crate-type; maturin handles this when building Python wheels
- `OnnxModel::load()`, `save_quantized()`, `CalibrationDataset::from_numpy()`, and `Config::from_file()` now accept `impl AsRef<Path>` instead of `&str`
- `from_f32_per_channel()` no longer takes a `channel_axis` parameter (was always 0; the function only supports axis 0)
- `CalibrationDataset` no longer stores a redundant `num_samples` field; `len()` returns `samples.len()` directly
- `total_size_bytes()` reads initializer sizes from the protobuf directly instead of re-extracting all weights
- `build_scale_tensor` and `build_zero_point_tensor` accept slices (`&[f32]`, `&[i8]`) to support both scalar and per-channel tensors
- `build_dequantize_linear_node` accepts `axis: Option<usize>` for per-channel DequantizeLinear
- `--bits` argument is now validated by clap at parse time; invalid values produce a usage error instead of a manual `eprintln` + `process::exit`
- Unknown calibration method in CLI now returns an error instead of silently falling back to percentile

### Removed

- `QuantMode` enum (unused, superseded by `QuantConfig.bits`)
- `CalibrationMethod::name()` (replaced by `Display` trait)
- `CalibrationDataset::get_batch()` (dead code, never called)
- `QuantParams.min`/`max` fields (stored but never read)
- `errors.rs` removed from public exports (module kept internal)
- Unused dependencies: `image`, `indicatif`, `serde_json`, `prost`

## [0.3.0] - 2026-02-04

### Added

- Python bindings via PyO3: `quantize()`, `quantize_with_calibration()`, `model_info()`
- `ActivationEstimator` -- runs real inference through tract to collect per-layer activation statistics
- Auto-detection of model input shapes for random calibration sample generation
- `ModelInfo` Python class exposing model name, version, node count, inputs, and outputs
- Graph connectivity validation (`OnnxModel::validate_connectivity`)
- `DequantizeLinear` QDQ graph transform for ONNX Runtime compatibility
- `activation_calibration` example demonstrating the full calibration pipeline
- `pyproject.toml` and maturin build configuration for Python wheel generation
- `python` feature flag in `Cargo.toml`

### Changed

- ONNX save now inserts `DequantizeLinear` nodes instead of renaming initializers; downstream graph nodes are unchanged
- Graph inputs that duplicate quantized initializer names are removed to avoid ONNX Runtime "duplicate definition" errors
- Opset version is bumped to >= 13 when saving quantized models (required by `DequantizeLinear` per-channel)

### Fixed

- ONNX Runtime refused to load quantized models because renamed initializers broke graph connectivity
- Percentile calibration incorrectly clipped at the lower bound

## [0.2.0] - 2026-01-15

### Added

- Per-channel quantization (separate scale/zero_point per output channel)
- INT4 quantization with bit-packing (two values per byte)
- Calibration framework with four methods: MinMax, Percentile, Entropy (KL divergence), MSE
- CLI commands: `batch`, `calibrate`, `validate`, `benchmark`, `config`
- YAML and TOML configuration file support
- 30+ unit tests

### Changed

- Improved error handling with `thiserror` and `anyhow`
- Progress bars via `indicatif` for long-running operations

### Fixed

- Shape mismatch errors in per-channel quantization for non-square tensors

## [0.1.0] - 2025-12-20

### Added

- Initial release
- INT8 per-tensor quantization for ONNX models
- CLI with `quantize` and `info` commands
- Weight extraction from ONNX protobuf initializers
- Quantized model saving

## Links

- PyPI: https://pypi.org/project/quantization-rs/
- Crates.io: https://crates.io/crates/quantize-rs
- Documentation: https://docs.rs/quantize-rs
- Repository: https://github.com/AR-Kamal/quantize-rs
