# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-06-06

This release is the pre-1.0 hardening pass: every change targets correctness,
robustness, or API stability so the next minor bump (v0.9.0) can soak before
being promoted to v1.0.

### API freeze & toolchain — final pre-1.0 pass

- **MSRV is now Rust 1.88.**  Declared via `rust-version` in `Cargo.toml`, with a
  dedicated CI job that builds on the pinned toolchain (read from the manifest,
  so the pin can't drift).  The floor is set by the `time` crate — a
  build-dependency pulled in transitively through the default `calibration`
  feature.
- **BREAKING:** the low-level, `prost`-coupled items are no longer part of the
  public API.  The free functions `apply_qdq_transform`,
  `apply_qdq_transform_with_options`, `validate_graph_connectivity`,
  `ensure_opset_version` and the entire `onnx_utils::quantization_nodes` module
  (`build_*` builders, `DequantLinearNames`, `StorageFormat`) are now
  crate-private.  They exposed `prost`-generated ONNX types (`GraphProto`,
  `TensorProto`, …) that were never meant to be stable — keeping them public
  would have made the `prost` version part of the semver contract.  Use the
  `OnnxModel` methods (`save_quantized` / `save_quantized_with_options` /
  `validate_connectivity`) instead; they were already the recommended path.
- **BREAKING:** `QuantRange` is now a **sealed** trait — it cannot be
  implemented outside the crate (it is internal INT8/INT4 machinery).  The
  public aliases `QuantParams`, `QuantParamsInt4`, `QuantizedTensor`, and
  `QuantizedTensorInt4` are unchanged.
- `QdqWeightInput` now implements `Default`, so the documented
  `QdqWeightInput { /* fields */, ..Default::default() }` construction compiles.
- The QDQ transform's post-pre-flight shape lookup returns a `GraphTransform`
  error instead of an internal `expect` panic, hardening it against a future
  refactor that could break the invariant.
- **Infrastructure:** the tag-triggered release workflow now runs a single
  `verify` gate (tag ↔ `Cargo.toml` ↔ `pyproject.toml` version check + tests +
  `cargo publish --dry-run`) before any publish, publishes crates.io before
  PyPI, and creates the GitHub release only when both registries succeed —
  shrinking the window for a partial/split release.

### Fixed — third-pass hardening

- **CLI `calibrate --method percentile:NN` is accepted** (e.g.
  `percentile:95`).  The clap validator now delegates to
  `CalibrationMethod::from_str`, which already supported the form, so the CLI
  and library can no longer disagree.  Previously the CLI rejected the
  documented syntax at parse time.
- **Re-quantizing an already-quantized model no longer corrupts it.**
  `OnnxModel::extract_weights` skips QDQ scale scaffolding (a `{base}_scale`
  FP32 initializer with a sibling `{base}_quantized`) and `_quantize_rs_`
  internal initializers, so a second `quantize` pass can't quantize the scales
  and silently break dequantization.
- **`save_quantized` cleans up the temp file on a write/fsync failure**, not
  only on rename failure — a failed save never leaves an orphan
  `.quantize-rs.tmp`.
- **Calibration `.npy` data in Fortran (column-major) order is read in the
  correct sample order** by forcing a standard layout before slicing; the
  common C-contiguous case stays zero-copy.
- **`validate --detailed` can no longer panic on a long non-ASCII weight
  name** — the display truncation counts characters, not bytes.
- **`examples/activation_calibration` input-shape auto-detection works.**  It
  now reads the model's declared input shapes via `OnnxModel::input_shapes()`
  instead of parsing input *name* strings (which carry no shape), so it no
  longer always falls back to `[3, 224, 224]`.

### Changed — third-pass hardening

- **BREAKING:** CLI `validate` exits non-zero when validation fails, so it can
  gate a CI pipeline.  JSON output still carries `validation_passed`.
- **BREAKING:** `ModelInfo` is now `#[non_exhaustive]` and gains an
  `opset_version` field — the default-domain opset that governs operator
  compatibility (usually more useful than the often-zero `model_version`).
  Surfaced in CLI `info` (human + `--format json`) and Python `model_info`.
- `quantize` runs graph-connectivity validation after saving (parity with
  `calibrate`), not just a reload.
- `calibrate` and both Python functions report a clear "all layers filtered
  out" error instead of the internal empty-slice save message; Python
  `quantize` now emits the FP16/BF16 "convert to FP32 first" hint on a
  non-FP32 model.
- Conflicting `--layer-bits NAME=…` for the same layer warns on stderr
  (last value wins).
- The atomic-save temp file uses a unique per-process name (`pid` + counter)
  so concurrent saves to the same output path can't collide.

### Added — third-pass hardening

- Nine new regression tests covering: CLI `percentile:NN` parsing,
  `--layer-bits` conflict handling, QDQ-scaffolding exclusion in
  `extract_weights`, no-orphan-`.tmp` on a failed save,
  `ModelInfo::opset_version`, and Fortran-order `.npy` slicing.

### Fixed — fourth-pass hardening

- **BatchNorm parameters and biases are no longer quantized.**  `extract_weights`
  now returns only rank-≥2 tensors.  It previously quantized *every* FP32
  initializer, including 1-D BatchNorm `scale`/`B`/`mean`/`var` and biases — and
  per-tensor INT8 on a near-zero `running_var` rounds it to 0, so the
  `1/sqrt(var)` in BatchNorm explodes the activations.  This broke any model with
  un-fused BatchNorm: MobileNetV2 went from cosine ≈ 0.10 to **0.978** (INT8) /
  **0.997** (per-channel) against the FP32 reference in ONNX Runtime.
- **`ActivationStats::update` no longer panics on a zero-count stats object.**
  Updating a `default()` / empty- / all-NaN-seeded `ActivationStats` with
  all-zero data kept the merged range at `[0, 0]`, which skipped histogram
  allocation and then indexed an empty histogram.  `update` now bootstraps from
  the first batch when `count == 0`.
- **External-data models get a clear, actionable error instead of "no weights
  found".**  Initializers whose data lives in a sidecar file
  (`data_location == EXTERNAL`, common for >2 GB exports) are detected and
  reported with instructions to re-save with weights embedded — in both the CLI
  and Python `quantize`.

### Changed — fourth-pass hardening

- **Saving a model that carries ONNX sections quantize-rs does not preserve now
  warns on stderr.**  The vendored protobuf schema is a subset, so re-encoding
  drops `ModelProto.functions` (local-function custom ops),
  `GraphProto.sparse_initializer`, and `ModelProto.training_info`.  Their
  presence is detected at load time (via a minimal probe schema) and a warning
  is printed on save, so a model relying on them isn't silently corrupted.
- **Bumping a model's opset across a breaking operator boundary now warns about
  ops quantize-rs cannot auto-migrate** (e.g. `Slice`, `Resize`, `Pad`,
  `Squeeze`, whose attributes moved to inputs).  The saved model would otherwise
  declare the new opset while keeping the old node form — which ONNX Runtime may
  reject; the warning points at `onnx.version_converter`.
- **`ir_version` is raised in lock-step with the opset.**  Native INT4 emits
  opset 21, whose `INT4`/`UINT4` types require IR ≥ 10; the model previously kept
  its source `ir_version` (e.g. 8), which is out of spec even though current ORT
  tolerates it.  `ensure_opset_version` now bumps `ir_version` to the spec
  minimum for the opset it sets.
- **Python packaging:** `requires-python` is now `>=3.9` (the abi3-py39 wheel
  can't load on 3.8 anyway), and the `.pyi`/`py.typed` stubs are bundled into the
  sdist as well as the wheel (`include` `format = "all"`).
- **Library warnings now go through the `log` facade instead of `eprintln!`.**
  The dropped-section, opset-migration, and temp-file-cleanup warnings are
  emitted with `log::warn!`, so library and Python consumers can route or
  suppress them rather than having stderr written from under them.  The CLI
  installs a small stderr logger so warnings still print there as `warning: …`;
  the Python bindings install `pyo3-log` so warnings flow into Python's
  `logging`.  Embedders that install no logger get no warning output — the
  standard Rust library contract.

- **`Quantizer::quantize_weights(&[WeightTensor])`** — quantize
  already-extracted weights without re-extracting.  The CLI `quantize` /
  `calibrate` and Python `quantize` paths now extract weights once instead of
  twice (they previously decoded every initializer's `raw_data` into f32 a
  second time inside `quantize_model`).
- **PEP 561 type stubs (`quantize_rs.pyi` + `py.typed`)** ship in the wheel, so
  editors / mypy / pyright get completion and type checking for the Python API.
- **ONNX Runtime execution smoke test** (`eval/ort_smoke_test.py`, run in CI):
  builds a tiny Conv model, quantizes it (per-tensor INT8, per-channel INT8,
  INT4 widened, and native INT4 opset 21), and confirms each result loads and
  runs in ONNX Runtime — closing the gap between structural graph checks and
  real ORT execution.
- **Real-model eval suite reconciled with reality.**  `eval/validate_models.py`
  had MobileNetV2's input name wrong (`"input"` vs `"data"`), so its FP32
  baseline never ran; fixed.  Per-tensor INT4 on MobileNetV2's depthwise convs
  is now tracked as an expected limitation (`XFAIL`) rather than failing the
  suite.  **8 of 9 configurations pass** (3 models × 3 configs).
- Eleven new tests across this pass: `update` zero-count guard (×3),
  dropped-section probe tags (×3), opset-migration detector, `ir_version` bump
  (×2), rank-1 BatchNorm/bias exclusion, and external-data skip.

### Fixed — second-pass audit (D1–D7)

- **Calibration auto-shape detection works for symbolic batch dimensions.**
  The CLI `calibrate` and Python `quantize_with_calibration` paths strip the
  batch slot (first dim) *before* filtering out symbolic / non-positive
  dims.  Previously, for the typical HuggingFace export shape
  `[batch, 3, 224, 224]` where `batch` is `dim_param` (returned as `-1`),
  the old logic filtered out `-1` first and then stripped what it thought
  was the batch — actually the channel — yielding the wrong `[224, 224]`
  sample shape and either tract failures or garbage activation stats.  Now
  yields the correct `[3, 224, 224]`.
- **`OnnxModel::from_bytes` enforces the 10 GB cap.**  Previously
  `from_bytes` skipped the size guard that `load` applies, leaving callers
  feeding bytes from HTTP, IPC, or fuzz harnesses exposed to OOM on
  pathological length-prefixed protobufs.  `MAX_MODEL_SIZE_BYTES` is now a
  module-level constant used by both code paths.
- **`save_quantized_with_options` with an empty input slice now errors
  instead of silently bumping the opset and wiping `quantize_rs.bits.*`
  metadata.**  Repeated saves of an already-quantized model with `&[]`
  would otherwise delete the bit-width metadata.  Callers who want to
  serialize the proto unchanged should encode and write it directly.
- **`from_bytes` error message no longer prints empty-quote path.**  The
  `QuantizeError::ModelLoad`/`ModelSave` Display now special-cases an
  empty `PathBuf` so the error reads `failed to load model (from bytes):`
  instead of `failed to load model '':`.
- **`count_non_fp32_weight_initializers` no longer counts INT64 shape
  constants.**  The heuristic is restricted to float-family dtypes
  (FP16/BF16/Double) — the only dtypes that could plausibly be weights —
  so the CLI's "model has N non-FP32 weight initializers — convert to FP32
  first" message doesn't trigger on models with rank-≥2 `Reshape`/`Tile`
  shape inputs.
- **Per-channel axis=0 limitation is now documented at every entry point.**
  CLI `--per-channel` help text and Python docstrings call out that
  per-channel quantization always operates on axis 0 (the Conv/MatMul
  output-channel dim); Transformer-style axis=1 layers are explicitly
  noted as unsupported.  Behavior is unchanged.
- **Orphan `.quantize-rs.tmp` cleanup failures are now logged.**  If the
  atomic-save rename fails AND the best-effort cleanup also fails (file
  locked by antivirus, EACCES, etc.), the path is printed to stderr so
  the user knows there's a stray file to delete manually.

### Added — second-pass audit (D8)

- **Nine new regression tests covering pre-1.0 behaviors** in
  `tests/integration.rs`:
  - `test_resave_reports_already_quantized` — second save on the same
    `OnnxModel` instance reports "already been quantized" diagnostic.
  - `test_save_quantized_no_orphan_tmp_after_success` — atomic save's
    `.quantize-rs.tmp` sentinel is cleaned up on success.
  - `test_save_quantized_with_empty_inputs_errors` — empty input slice
    is rejected.
  - `test_from_bytes_rejects_oversize_input` — size-cap branch is wired.
  - `test_quantized_weight_info_option_accessors_happy_path` — `scale()`
    / `zero_point()` return `Some(_)` on a real quantized model.
  - `test_count_non_fp32_weight_initializers_filters_int_shape_tensors`
    — INT64 rank-2 shape constants and rank-1 biases are excluded.
  - `test_batch_config_jobs_parses_from_yaml` — `BatchConfig.jobs`
    round-trips through YAML.
  - `test_batch_config_jobs_defaults_to_one` — default fallback.
  - `test_from_bytes_error_message_omits_empty_path` — Display tweak.

### Changed — CI breadth (D9)

- Clippy now runs on the full OS matrix (Ubuntu / Windows / macOS) instead
  of Ubuntu-only.  Platform-specific code paths (`std::fs::rename` for the
  atomic save, mmap, path handling) are linted on every supported OS.

### Fixed (correctness — blockers for v1.0)

- **CLI exit codes propagate failure.** `quantize`, `batch`, and `config`
  now return `Err` (non-zero exit) on:
  - Empty / malformed input (model loaded but contains 0 weight tensors).
  - `batch` with a glob pattern that matches no files.
  - `batch` where at least one model failed (in serial mode this was already
    correct via `--continue-on-error`; in `--jobs > 1` mode it was previously
    a silent success).
  - `config` run where any individual model failed (was previously a silent
    success, returning `Ok(())` after printing per-model error lines).
  When the FP32 weight set is empty but the model contains non-FP32
  weight-shaped initializers (FP16/BF16/INT8), the error message now says so
  explicitly — "convert the model to FP32 first" instead of a generic
  "no weights found".
- **`apply_qdq_transform_with_options` fails fast and clean on retry.**
  A pre-flight pass verifies every requested weight name still exists as an
  FP32 initializer before any mutation.  If the user calls `save_quantized`
  twice on the same `OnnxModel`, the second call now reports
  "Weight 'X' has already been quantized in this graph (found 'X_quantized'
  initializer); apply_qdq_transform is not idempotent — load a fresh
  OnnxModel before retrying" instead of leaving the graph half-mutated.
- **`save_quantized` writes atomically.** Output is now written to a sibling
  `<path>.quantize-rs.tmp`, `fsync`ed, and renamed into place.  A crash or
  power loss mid-write can no longer leave a torn/truncated `.onnx` at the
  target path; readers see either the old file or the fully-written new one.
- **`metadata_props` deduplicated.** Repeated `save_quantized_with_options`
  calls on the same `OnnxModel` instance no longer accumulate duplicate
  `quantize_rs.bits.<name>` entries; the prior set is dropped before each
  save emits the current one.
- **`batch --jobs N` survives a poisoned stdout mutex.** Worker `println!`
  panics (broken pipe, etc.) no longer cascade into "poisoned Mutex" panics
  in sibling workers; the guard is recovered via `into_inner`.
- **`ActivationStats::update` mean precision.** The incremental mean is
  computed entirely in `f64` before casting once to `f32` at the end (the
  prior form cast inside the expression and lost precision when `old_count`
  was large).

### Changed (API surface for v1.0 stability)

- **BREAKING:** `QuantizedWeightInfo::scale()` and `zero_point()` now return
  `Option<f32>` / `Option<i8>` instead of panicking when the per-channel vec
  is empty.  Library-produced `QuantizedWeightInfo` is never empty in
  practice; this only matters if you construct one by hand.  Per-channel
  consumers should iterate the `scales` / `zero_points` slice fields
  directly.
- **BREAKING:** Public enums now carry `#[non_exhaustive]`:
  `QuantizeError`, `QuantizedTensorType`, `StorageFormat`,
  `CalibrationMethod`.  Downstream `match` arms must include a `_ => ...`
  catch-all so adding a future variant is a non-breaking change.
- **BREAKING:** Result-only public structs now carry `#[non_exhaustive]`:
  `SaveOptions`, `QuantizedWeightInfo`, `QuantizedWeightOutput`.  These
  types are returned by the library; external code constructs them via the
  documented constructors / accessors only.
- **BREAKING:** `ActivationEstimator::get_layer_stats_mut` removed.  Nothing
  in-tree used it; exposing a `&mut HashMap` over internal calibration
  state on a v1.0-stable API is risky.  Use `into_layer_stats()` (consume
  and own) or `get_layer_stats()` (borrow read-only) instead.
- **`CalibrationMethod::Display` now round-trips through `FromStr`.**
  Variants are rendered in their lowercase keyword form (`"minmax"`,
  `"entropy"`, `"mse"`).  `Percentile(p)` renders as `"percentile:p"` so the
  percentile value survives the round trip.  `FromStr` accepts the same set
  plus the back-compat `"percentile"` form (defaults to 99.9th percentile).
  The CLI `Method: …` log line and any tooling that parses calibration
  method strings sees the lowercase form.

### Added

- **`OnnxModel::count_non_fp32_weight_initializers()`** — returns the count
  of rank-≥2 initializers whose dtype is not FP32.  Surfaces FP16/BF16
  model detection in user-facing errors (see the FP32-only error message
  improvement above).
- **`ActivationEstimator::calibrate_quiet`** — same as `calibrate` but emits
  no `println!` output.  Used by the Python bindings so embedded callers
  don't get stdout noise.
- **`BatchConfig.jobs: usize`** — new field on the YAML/TOML batch config
  with `#[serde(default)]` (defaults to 1).  `quantize-rs config <file>`
  now honours it when invoking the parallel batch path; previously the
  config path was hardcoded to `jobs = 1`.
- **Python bindings release the GIL.**  `quantize` and
  `quantize_with_calibration` wrap their I/O- and CPU-heavy work in
  `py.allow_threads(...)` so other Python threads (and asyncio tasks) can
  run concurrently while quantization runs in Rust.
- **Python wheels build with PyO3 abi3-py39.**  `Cargo.toml` enables
  `pyo3/abi3-py39`; one wheel now covers Python 3.9 through 3.13+.  The
  release workflow drops the `--interpreter python3.12` pin.
- **`CalibrationMethod::Percentile(p)` parses from `"percentile:NN"`.**
  E.g. `"percentile:95"` → `Percentile(95.0)`.  Percentiles outside
  `[0, 100]` are rejected at parse time.  4 new unit tests cover the
  round-trip, the back-compat `"percentile"` form, invalid inputs, and
  discriminant identity for the keyword variants.
- **CI hardened.** `.github/workflows/ci.yml` now runs the test suite at
  three feature combinations (default, `--all-features`,
  `--no-default-features --lib`), runs clippy with
  `--all-features --all-targets` (so test-only lints are caught), and
  has a new `doc` job that builds rustdoc with `RUSTDOCFLAGS=-D warnings`.
  `.github/workflows/release.yml` gates `cargo publish` on the same
  test + clippy + fmt + doc checks.
- **CLI `calibrate` validates the saved model.**  Matches the existing
  `quantize` UX: reloads the output file and runs `validate_connectivity`,
  reporting any dangling edges before exit.

### Polish

- The Dropout opset-12 ratio→input upgrade in `ensure_opset_version` now
  embeds the node index in the synthesized initializer name
  (`_quantize_rs_dropout_ratio_<idx>_<first_output>`) so two Dropouts that
  happen to share a first-output name can't collide on the new initializer.
- The CLI smoke-tests its built binary against `mnist.onnx`:
  INT8 per-channel symmetric and native INT4 (opset 21) both produce
  ORT-loadable QDQ output with the documented 4× / 8× weight compression.
- `examples/activation_calibration.rs` is gated with
  `required-features = ["calibration"]` so `cargo check --no-default-features
  --examples` compiles cleanly.

### Test counts

- **Default features**: 109 unit + 4 bin + 39 integration + 17 property-based =
  **169 passing**, 5 ignored (up from 136 in v0.8.0).
- **`--all-features`**: 111 unit + 4 bin + 40 integration + 17 property-based =
  **172 passing**, 5 ignored (up from 139 in v0.8.0).

## [0.8.0] - 2026-04-25

### Added

- **Native INT4 storage** (`SaveOptions::with_native_int4(true)`, CLI `--native-int4`, Python `native_int4=True`): writes INT4 weights as ONNX `DataType::Int4` (opset 21) instead of widening to INT8 bytes. Gives true 8x on-disk compression. Has no effect on INT8-only models. ONNX wire-format INT4 packing puts the even index in the low nibble (distinct from the internal `pack_int4` layout, which puts `val1` in the high nibble); both round-trip cleanly through `pack_int4_onnx` / `unpack_int4_onnx`.
- **Symmetric quantization** (`QuantConfig.symmetric`, CLI `--symmetric`, Python `symmetric=True`): forces `zero_point == 0` and uses a balanced range `[-|max|, +|max|]`. Required by most ONNX Runtime / TensorRT INT8 matmul kernels for per-channel weight quantization. New constructors: `QuantParams::from_range_symmetric`, `from_f32_symmetric`, `from_f32_per_channel_symmetric`, `from_f32_with_range_symmetric`.
- **`Quantizer::quantize_model(&self, model: &OnnxModel) -> Result<Vec<QuantizedWeightOutput>>`**: consolidated entry point that runs the full filter / parallel / `layer_bits` pipeline. CLI `quantize` / `batch` / `calibrate` and both Python functions now route through it, eliminating ~200 lines of duplicated logic and fixing a bug where the Python calibration path silently skipped `excluded_layers`, `min_elements`, and `layer_bits`.
- **`QuantizedWeightOutput { qdq, quantized_size_bytes, mse }`** struct returned from `quantize_model`.
- **`SaveOptions { native_int4: bool }`** with builder methods (`with_native_int4`). New `OnnxModel::save_quantized_with_options(...)` and `apply_qdq_transform_with_options(...)`. Existing `save_quantized` / `apply_qdq_transform` remain as thin wrappers.
- **Memory-mapped model loading** (feature `mmap`, dependency `memmap2`): `OnnxModel::load_mmap()` streams multi-GB ONNX files without copying the full bytes into RAM.
- **safetensors calibration input** (feature `safetensors-input`, dependency `safetensors`): `CalibrationDataset::from_safetensors`, `from_safetensors_named` -- load calibration tensors directly from HuggingFace `.safetensors` files.
- **JSON output** for `validate` / `info` / `benchmark` CLI commands via `--format json`. New `ValidateReport`, `InfoReport`, `BenchmarkReport` serde structs. Banner is suppressed in JSON mode so stdout is parseable.
- **Parallel batch processing**: `quantize-rs batch ... --jobs N` runs N model conversions concurrently via a rayon `ThreadPoolBuilder`, with stdout serialized through a `Mutex` to keep progress lines intact.
- **Histogram-direct calibration optimization** (`calculate_optimal_range_from_stats`, `histogram_kl_divergence`, `histogram_quantization_mse`, `optimize_kl_from_stats`, `optimize_mse_from_stats`): KL/MSE range search now reads the histogram directly instead of regenerating samples through an RNG. Deterministic and ~2x faster on the calibration path.
- **`QuantizedWeightInfo::storage_bytes`**: actual byte count from `raw_data.len()`, replacing the previous heuristic. Fixes `benchmark`'s weight-compression ratio reporting `4.0` for native-INT4 models that should be `8.0`.
- **Filter flags on `calibrate`**: `--exclude`, `--min-elements`, `--layer-bits`, `--native-int4`, `--symmetric` now available on the CLI calibrate subcommand (previously only on `quantize` / `batch`).
- **Fuzz target** (`fuzz/`): standalone `cargo-fuzz` workspace with one `onnx_load` target stress-testing `OnnxModel::from_bytes`. See `fuzz/README.md`.
- **`OnnxModel::from_bytes(&[u8])`**: in-memory load, used by the fuzz target and useful for processing models from non-file sources.
- ~25 new integration tests covering native INT4, symmetric round-trip, mmap parity, safetensors loading, JSON output schemas, and benchmark structure-preservation. Total test count: **136 with default features** (95 unit + 24 integration + 17 property-based), **139 with `--all-features`**.

### Changed

- **Performance:** ~25% measured speedup on INT4 per-channel quantization. Replaces per-channel `Vec` allocation with `chunks_exact` slice iteration in the hot loop; removes a per-element division in `to_f32` dequantization.
- `validate` and `benchmark` use a QDQ-aware structure check: the expected post-transform node count is `original_nodes + DequantizeLinear_count`, and the input set may grow by the number of new quantized initializers exposed as graph inputs. The previous strict equality check reported false negatives on every valid QDQ model.
- `batch` command's terminal banner now distinguishes "all skipped" (when `--skip-existing` matched everything) from "all succeeded" -- previously both printed the same green check.
- Python bindings (`quantize`, `quantize_with_calibration`) accept `native_int4=False` and `symmetric=False` kwargs. Both functions now share the same `Quantizer::quantize_model` pipeline (the calibration path previously bypassed filtering and `layer_bits`).
- `QuantConfig` gains `symmetric: bool`; existing struct-literal callers need `..Default::default()` on the trailing fields.

### Fixed

- **Per-channel scale/zero_point round-trip:** `QuantizedWeightInfo` previously stored single `scale: f32` / `zero_point: i8` fields, silently dropping per-channel data on the read-back path. Now stores `scales: Vec<f32>` / `zero_points: Vec<i8>` with `scale()` / `zero_point()` accessors and `is_per_channel()`. **BREAKING** for direct field access; the `scale()` / `zero_point()` methods provide the previous shape for per-tensor consumers.
- **Native-INT4 byte count:** `benchmark` now reports the correct on-disk size (and 8x compression ratio) for `--native-int4` outputs by reading `storage_bytes` from `raw_data.len()`.
- `QdqWeightInput` now derives `Clone` (required by the `quantize_model` -> `QuantizedWeightOutput` pipeline).

### Removed

- `sample_from_activation_stats` and `extract_channel` (dead after the histogram-direct rewrite).

## [0.7.0] - 2026-03-28

### Fixed

- **NaN undefined behaviour**: `QuantParams::from_range()` now guards against NaN before the `as i8` cast. Previously, NaN from degenerate inputs (e.g., all-zero tensors with `min == max`) caused undefined behaviour on the integer cast.
- **Constant-value tensor quantization**: Changed the `(min ± 0.01)` hack to a symmetric range `(-|v|, +|v|)`, producing accurate dequantization for constant tensors instead of the previous off-by-0.01 error.
- **`chunks_exact(4)` panic**: `extract_weights()` now validates that `raw_data.len() % 4 == 0` before chunking, skipping misaligned initializers instead of panicking.
- **Calibration NaN/Inf poisoning**: `ActivationStats::update()` now filters non-finite values before computing statistics, preventing backwards-infinity min/max bounds.
- **MSE calibration clamp**: MSE error computation now clamps the quantized value to `[0, 255]` before computing the error, fixing incorrect range optimization.
- **Opset bump breaks old models**: `ensure_opset_version()` previously bumped all models to opset 13 without upgrading deprecated op attributes. Now:
  - Only bumps to minimum needed opset (10 for per-tensor, 13 for per-channel DequantizeLinear).
  - Strips `BatchNormalization.spatial` attribute when bumping past opset 9.
  - Migrates `Dropout.ratio` from attribute to input when bumping past opset 12.
  - Adds explicit `axis=1` to `Softmax`/`LogSoftmax` nodes when bumping past opset 13 (default changed from 1 to -1).
- **All-weights-excluded silent no-op**: CLI `quantize` now warns when all tensors are excluded instead of silently producing an unchanged model.
- **Empty `layer_bits` key**: Config validation now rejects empty strings as layer names in `layer_bits`.
- **Range calculation deduplication**: Extracted `finite_min_max()` helper, eliminating duplicated min/max computation in the calibrated quantization path.
- **`cargo fmt`**: Fixed pre-existing formatting issues across the entire codebase.

### Added

- **CI pipeline** (`.github/workflows/ci.yml`): 4 jobs — test matrix (Ubuntu/Windows/macOS), clippy (`-D warnings`), `cargo fmt --check`, benchmarks compile check.
- **Real-world model validation** (`eval/validate_models.py`): Downloads ONNX models (ResNet-18, SqueezeNet), quantizes with quantize-rs CLI, loads in ONNX Runtime, compares FP32 vs quantized outputs (cosine similarity, max error, top-K match). 6/6 configurations pass.
- **Opset upgrade engine** in `graph_builder.rs`: `upgrade_deprecated_ops()` handles breaking attribute changes across opset boundaries (BatchNormalization, Dropout, Softmax).
- 5 new calibration integration tests: `test_calibrated_quantization_uses_stats`, `test_calibrated_quantization_with_method`, `test_calibrated_quantization_fallback_no_stats`, `test_calibrated_quantization_int4`, `test_calibrated_full_pipeline`.
- 2 new opset upgrade unit tests: `test_ensure_opset_strips_deprecated_attrs`, `test_ensure_opset_no_downgrade`.
- 1 new config validation unit test: `test_empty_layer_bits_key_rejected`.
- Total test count: 106 passing (71 unit + 18 integration + 17 property-based), 5 ignored.

### Changed

- `save_quantized()` now computes minimum required opset (10 for per-tensor, 13 for per-channel) instead of always bumping to 13.

## [0.6.0] - 2026-02-19

### Added

- **Dependency modernization**: Replaced the unmaintained `onnx = "0.1.0"` crate (which depended on `protobuf` v1.7) with a prost-based protobuf pipeline. The ONNX schema is now vendored as `proto/onnx.proto3`; `build.rs` compiles it at build time using `protox` (pure-Rust protoc replacement) + `prost-build`. No system `protoc` binary required — CI no longer needs to install it.
- **Per-layer exclusion**: `QuantConfig.excluded_layers: Vec<String>` — layer names listed here are left in FP32 and skipped during quantization.
- **Per-layer bit-width overrides**: `QuantConfig.layer_bits: HashMap<String, u8>` — individual layers can be quantized to a different bit width than the global default.
- **Minimum-elements threshold**: `QuantConfig.min_elements: usize` — tensors with fewer elements than this value (e.g., biases) are kept in FP32.
- `QuantConfig::should_quantize(name, num_elements) -> bool` helper encapsulating both the exclude-list and min-elements checks.
- `QuantConfig::bits_for_layer(name) -> u8` helper returning the per-layer or global bit width.
- CLI `quantize` subcommand gains two new flags:
  - `--exclude <LAYER>` (repeatable) — exclude a layer by name
  - `--min-elements <N>` — skip tensors with fewer than N elements
- `Config.excluded_layers` and `Config.min_elements` global fields in YAML/TOML config files.
- `ModelConfig.excluded_layers`, `ModelConfig.layer_bits`, `ModelConfig.min_elements` per-model overrides in config files.
- `Config::get_excluded_layers()` and `Config::get_min_elements()` helpers (model overrides merged with global).
- 6 new unit tests for `should_quantize` and `bits_for_layer` in `quantization/mod.rs`.
- **Property-based tests** (`tests/property_tests.rs`, 15 tests using `proptest`):
  - `quantize(dequantize(v)) ≈ v` for exact grid points (INT8 and INT4)
  - `|dequantize(quantize(v)) - v| ≤ scale/2` for all values in range
  - `pack(unpack(data)) == data` for all valid INT4 data
  - Quantization never panics on any finite f32 input
  - Per-channel quantization error bounds
- **Criterion benchmarks** (`benches/quantization.rs`, 4 groups):
  - `quantize_throughput` — INT8/INT4 at 1K/100K/1M elements with `Throughput::Elements`
  - `per_channel_vs_per_tensor` — comparison across 4 variants on a [64,27] tensor
  - `pack_int4` — raw pack/unpack throughput at 10K/100K/1M elements
  - `quantize_model` — full Quantizer loop over 8 synthetic weight tensors
- **`validate_real_model` example** (`examples/validate_real_model.rs`): loads any ONNX file, quantizes weights, reports per-tensor MAE and compression, and optionally saves + validates the quantized model. Accepts `--bits`, `--per-channel`, `--min-elements`, `--output`.
- **GPT-2 evaluation script** (`eval/benchmark_gpt2.py`): end-to-end benchmark comparing FP32 vs INT8 GPT-2 small. Three-step workflow: `--export` (HuggingFace → ONNX), `--quantize` (calls `validate_real_model` binary), `--benchmark` (perplexity + text generation). Validated results: −74.8% file size, +1.78% perplexity on WikiText-2 (negligible quality loss).
- Additional multilayer integration tests in `tests/integration.rs`: `test_multilayer_min_elements`, `test_multilayer_excluded_layers`, `test_multilayer_full_round_trip`, `test_multilayer_compression_ratio`.

### Changed

- `QuantConfig` now derives `Default`; existing struct-literal instantiations require `..Default::default()` for the new fields.
- `commands::quantize()` signature extended: `excluded_layers: &[String]`, `min_elements: usize`.
- Quantization loop in `commands::quantize()` filters weights with `config.should_quantize()` before parallelizing and applies per-layer bits via `config.bits_for_layer()`.
- CLI quantize output now prints `Quantized: N/M tensors` when layers are skipped.
- Total test count: 90 passing (63 unit + 12 integration + 15 property-based), 7 ignored (require model files on disk).

### Removed

- `onnx = "0.1.0"` and `protobuf = "1.7"` from `[dependencies]` — replaced by the prost pipeline above.

## [0.5.0] - 2026-02-18

### Added

- `QuantizeError` enum with 8 variants (`InvalidTensor`, `UnsupportedConfig`, `ModelLoad`, `ModelSave`, `GraphTransform`, `Calibration`, `Config`, `Other`) replacing `anyhow::Result` at all public API boundaries
- `errors::Result<T>` type alias for `std::result::Result<T, QuantizeError>`
- `pub mod errors` and `pub use errors::QuantizeError` in crate root
- `///` doc comments on all public items (structs, enums, methods, functions, modules)
- Crate-level `//!` documentation with module overview and feature flags
- 6 integration tests in `tests/integration.rs` that construct ONNX models in memory (no model files needed)
- `tempfile` dev-dependency for integration test I/O

### Changed

- **BREAKING:** All public library functions now return `crate::errors::Result<T>` instead of `anyhow::Result<T>`
- **BREAKING:** `CalibrationMethod::from_str` error type changed from `anyhow::Error` to `QuantizeError`
- CLI (`main.rs`) and Python bindings (`python.rs`) are unchanged in behavior -- `QuantizeError` auto-converts to `anyhow::Error` and `PyRuntimeError` respectively
- `anyhow` remains in `Cargo.toml` for CLI binary use

### Fixed

- `rustdoc` warning for unescaped `[num_channels]` in `graph_builder.rs`
- `rustdoc` warning for unescaped `Vec<i64>` in `onnx_utils/mod.rs`

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
