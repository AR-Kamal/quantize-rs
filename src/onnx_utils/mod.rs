// src/onnx_utils/mod.rs
//! ONNX model utilities — loading, weight extraction, quantized save (QDQ),
//! graph connectivity validation, and quantized-model introspection.

pub mod graph_builder;
// Internal QDQ node-construction helpers. Kept crate-private: they traffic in
// raw prost `onnx_proto` types (NodeProto/TensorProto), which are `#[doc(hidden)]`
// and not part of the stable public API. Use the `OnnxModel` save methods instead.
pub(crate) mod quantization_nodes;

use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{
    tensor_proto, tensor_shape_proto, type_proto, ModelProto, StringStringEntryProto,
};
use prost::Message;
use std::fs;
use std::io::{Read, Write};

// Re-export so callers don't have to reach into submodules
pub use graph_builder::{ConnectivityReport, SaveOptions};

/// Maximum accepted ONNX model size for [`OnnxModel::load`],
/// [`OnnxModel::load_mmap`], and [`OnnxModel::from_bytes`].
///
/// Inputs over this size are rejected to prevent OOM from pathological or
/// malicious protobufs.  10 GB is well above any production ONNX model in
/// the wild — including multi-billion-parameter LLMs — while still being a
/// hard ceiling that protects callers feeding bytes from untrusted sources
/// (HTTP, IPC, fuzz harnesses).
const MAX_MODEL_SIZE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

// ===========================================================================
// Core types
// ===========================================================================

/// An ONNX model loaded from a protobuf file.
///
/// Provides methods for inspecting, extracting weights, saving quantized
/// models, and validating graph connectivity.
pub struct OnnxModel {
    proto: ModelProto,
    /// Wire-format sections present in the *source* bytes that the vendored
    /// schema does not model and that will therefore be dropped when the model
    /// is re-encoded on save (see [`dropped_scan`]).  Populated at load time;
    /// `Default` (all-zero) for any other construction path.
    dropped: DroppedSections,
}

/// Count of ONNX wire-format sections that round-tripping through the vendored
/// (minimal) protobuf schema would silently drop on save.
///
/// `quantize-rs` models a subset of the ONNX schema, and prost discards fields
/// it doesn't know.  Most are inert metadata, but a few carry real graph
/// semantics — most importantly `ModelProto.functions` (local-function custom
/// ops).  We detect their presence at load time so the save path can warn
/// instead of silently writing a model that omits them.
#[derive(Debug, Clone, Default)]
struct DroppedSections {
    /// `ModelProto.functions` — local `FunctionProto` definitions.
    functions: usize,
    /// `GraphProto.sparse_initializer` — sparse weight tensors.
    sparse_initializers: usize,
    /// `ModelProto.training_info` — training metadata.
    training_info: usize,
}

impl DroppedSections {
    /// Re-decode the raw model bytes with a probe schema that models *only* the
    /// unmodeled tags, letting prost do the wire parsing.  A wrong tag here can
    /// only produce a spurious or missing warning — never corruption — because
    /// nothing is written from the result.
    fn scan(bytes: &[u8]) -> Self {
        use prost::Message;
        match dropped_scan::ProbeModel::decode(bytes) {
            Ok(p) => Self {
                functions: p.functions.len(),
                sparse_initializers: p.graph.map(|g| g.sparse_initializer.len()).unwrap_or(0),
                training_info: p.training_info.len(),
            },
            // The bytes already decoded once as a full ModelProto, so a probe
            // failure is unexpected; treat it as "nothing dropped".
            Err(_) => Self::default(),
        }
    }

    fn any(&self) -> bool {
        self.functions > 0 || self.sparse_initializers > 0 || self.training_info > 0
    }

    /// Human-readable list of what will be dropped, for the save-time warning.
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.functions > 0 {
            parts.push(format!("{} local function definition(s)", self.functions));
        }
        if self.sparse_initializers > 0 {
            parts.push(format!(
                "{} sparse initializer(s)",
                self.sparse_initializers
            ));
        }
        if self.training_info > 0 {
            parts.push(format!("{} training-info section(s)", self.training_info));
        }
        parts.join(", ")
    }
}

/// Minimal prost messages that model *only* the ONNX wire-format tags the main
/// schema intentionally omits.  Decoding a model's bytes a second time with
/// these tells us whether the round-trip will drop anything, without a
/// hand-rolled protobuf scanner.  Field numbers match the official ONNX spec
/// (`functions` = 25, `training_info` = 20, `GraphProto.sparse_initializer`
/// = 15); a `graph` is descended via its real tag (7).
mod dropped_scan {
    use prost::Message;

    /// Empty placeholder: prost decodes a sub-message into this and skips all
    /// of its contents, but each occurrence is still counted.
    #[derive(Clone, PartialEq, Message)]
    pub(super) struct Ignore {}

    #[derive(Clone, PartialEq, Message)]
    pub(super) struct ProbeGraph {
        #[prost(message, repeated, tag = "15")]
        pub sparse_initializer: Vec<Ignore>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub(super) struct ProbeModel {
        #[prost(message, optional, tag = "7")]
        pub graph: Option<ProbeGraph>,
        #[prost(message, repeated, tag = "20")]
        pub training_info: Vec<Ignore>,
        #[prost(message, repeated, tag = "25")]
        pub functions: Vec<Ignore>,
    }
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self
            .proto
            .graph
            .as_ref()
            .map(|g| g.name.as_str())
            .unwrap_or("");
        let num_nodes = self.proto.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);
        f.debug_struct("OnnxModel")
            .field("name", &name)
            .field("num_nodes", &num_nodes)
            .finish()
    }
}

/// Summary of an ONNX model's structure.
///
/// Marked `#[non_exhaustive]` so future summary fields can be added without a
/// breaking change.
#[derive(Debug)]
#[non_exhaustive]
pub struct ModelInfo {
    /// Graph name from the protobuf.
    pub name: String,
    /// `model_version` field from the protobuf (often 0 in practice).
    pub version: i64,
    /// Default-domain opset version the model declares (0 if absent).  This is
    /// the value that governs operator compatibility — usually more useful
    /// than [`version`](Self::version).
    pub opset_version: i64,
    /// Number of computation nodes in the graph.
    pub num_nodes: usize,
    /// Names of the graph inputs.
    pub inputs: Vec<String>,
    /// Names of the graph outputs.
    pub outputs: Vec<String>,
}

/// Metadata about a quantized weight recovered from a QDQ-format model.
///
/// Marked `#[non_exhaustive]` so future fields can be added without a
/// breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QuantizedWeightInfo {
    /// Original weight name (without `_quantized` suffix).
    pub name: String,
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Quantization scales.  `len() == 1` for per-tensor quantization;
    /// `len() == num_channels` for per-channel.
    pub scales: Vec<f32>,
    /// Quantization zero points.  Same length as [`scales`](Self::scales).
    pub zero_points: Vec<i8>,
    /// Number of elements in the quantized tensor.
    pub original_length: usize,
    /// Actual on-disk byte count of the quantized initializer's `raw_data`.
    /// For INT8 storage this equals `original_length`; for native INT4
    /// (opset 21) it is `ceil(original_length / 2)`.
    pub storage_bytes: usize,
}

impl QuantizedWeightInfo {
    /// `true` if the weight was quantized per-channel (more than one scale).
    pub fn is_per_channel(&self) -> bool {
        self.scales.len() > 1
    }

    /// Per-tensor convenience accessor: returns the first scale, or `None`
    /// if no scales were recovered for this weight (malformed model).
    ///
    /// For per-channel tensors, iterate over [`scales`](Self::scales) instead.
    pub fn scale(&self) -> Option<f32> {
        self.scales.first().copied()
    }

    /// Per-tensor convenience accessor: returns the first zero-point, or
    /// `None` if no zero-points were recovered (malformed model).
    ///
    /// For per-channel tensors, iterate over [`zero_points`](Self::zero_points) instead.
    pub fn zero_point(&self) -> Option<i8> {
        self.zero_points.first().copied()
    }
}

// ===========================================================================
// OnnxModel — load / inspect
// ===========================================================================

impl OnnxModel {
    /// Load an ONNX model from a file path.
    ///
    /// Reads the entire file into a `Vec<u8>` before decoding.  For
    /// multi-gigabyte models consider [`load_mmap`](Self::load_mmap)
    /// (requires the `mmap` feature) to avoid the extra heap buffer.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::ModelLoad`] if the file cannot be opened,
    /// is too large (>10 GB), or contains invalid protobuf data.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = fs::File::open(path).map_err(|e| QuantizeError::ModelLoad {
            path: path.to_path_buf(),
            reason: format!("Failed to open ONNX file: {e}"),
        })?;

        let file_size = file
            .metadata()
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to read metadata: {e}"),
            })?
            .len();
        if file_size > MAX_MODEL_SIZE_BYTES {
            return Err(QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!(
                    "Model file too large: {:.2} GB (max: 10 GB)",
                    file_size as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
            });
        }

        let mut buffer = Vec::with_capacity(file_size as usize);
        file.read_to_end(&mut buffer)
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to read ONNX file: {e}"),
            })?;

        let proto = ModelProto::decode(&buffer[..]).map_err(|e| QuantizeError::ModelLoad {
            path: path.to_path_buf(),
            reason: format!("Failed to parse ONNX protobuf: {e}"),
        })?;

        let dropped = DroppedSections::scan(&buffer);
        Ok(Self { proto, dropped })
    }

    /// Decode an ONNX model directly from a byte slice.
    ///
    /// Useful for in-memory or fuzzing scenarios where the source isn't a
    /// filesystem path.  The same 10 GB size cap that [`load`](Self::load)
    /// applies to files is also enforced here so callers feeding bytes from
    /// untrusted sources (HTTP, IPC, fuzz harnesses) can't OOM the decoder
    /// with a pathologically large input.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::ModelLoad`] if `bytes` exceeds 10 GB or
    /// cannot be decoded as a `ModelProto`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() as u64 > MAX_MODEL_SIZE_BYTES {
            return Err(QuantizeError::ModelLoad {
                path: std::path::PathBuf::new(),
                reason: format!(
                    "Input too large: {:.2} GB (max: 10 GB)",
                    bytes.len() as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
            });
        }
        let proto = ModelProto::decode(bytes).map_err(|e| QuantizeError::ModelLoad {
            path: std::path::PathBuf::new(),
            reason: format!("Failed to parse ONNX protobuf: {e}"),
        })?;
        let dropped = DroppedSections::scan(bytes);
        Ok(Self { proto, dropped })
    }

    /// Load an ONNX model by memory-mapping the file (requires the `mmap`
    /// feature).
    ///
    /// Compared to [`load`](Self::load), this avoids the intermediate
    /// `Vec<u8>` buffer — useful for multi-gigabyte models where doubling
    /// the working set during decode is a problem.  Peak RAM during load
    /// falls from roughly `2 × file_size` to `1 × file_size + mmap overhead`.
    ///
    /// # Safety
    ///
    /// Memory-mapping requires that the file is not modified for the
    /// duration of the load.  Another process truncating or rewriting the
    /// file while decoding would be undefined behaviour.  This function
    /// uses the `unsafe { Mmap::map(&file) }` call under the hood; its
    /// invariants are the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::ModelLoad`] on I/O failure, invalid size,
    /// or malformed protobuf.
    #[cfg(feature = "mmap")]
    pub fn load_mmap(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = fs::File::open(path).map_err(|e| QuantizeError::ModelLoad {
            path: path.to_path_buf(),
            reason: format!("Failed to open ONNX file: {e}"),
        })?;

        let file_size = file
            .metadata()
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to read metadata: {e}"),
            })?
            .len();
        if file_size > MAX_MODEL_SIZE_BYTES {
            return Err(QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!(
                    "Model file too large: {:.2} GB (max: 10 GB)",
                    file_size as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
            });
        }

        // SAFETY: see method-level docs — caller guarantees the file is
        // not modified while it is mapped.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to mmap ONNX file: {e}"),
            })?
        };

        let proto = ModelProto::decode(&mmap[..]).map_err(|e| QuantizeError::ModelLoad {
            path: path.to_path_buf(),
            reason: format!("Failed to parse ONNX protobuf: {e}"),
        })?;

        let dropped = DroppedSections::scan(&mmap[..]);

        // mmap is dropped here; `proto` owns all its data (prost copies
        // bytes out of the source during decode), so this is sound.
        Ok(Self { proto, dropped })
    }

    /// Return a summary of the model's structure.
    pub fn info(&self) -> ModelInfo {
        let graph = self.proto.graph.as_ref();

        let inputs: Vec<String> = graph
            .map(|g| g.input.iter().map(|i| i.name.clone()).collect())
            .unwrap_or_default();

        let outputs: Vec<String> = graph
            .map(|g| g.output.iter().map(|o| o.name.clone()).collect())
            .unwrap_or_default();

        // Default-domain opset (empty domain string) — what actually governs
        // operator compatibility.  `model_version` is usually 0 and unhelpful.
        let opset_version = self
            .proto
            .opset_import
            .iter()
            .find(|o| o.domain.is_empty())
            .map(|o| o.version)
            .unwrap_or(0);

        ModelInfo {
            name: graph.map(|g| g.name.clone()).unwrap_or_default(),
            version: self.proto.model_version,
            opset_version,
            num_nodes: graph.map(|g| g.node.len()).unwrap_or(0),
            inputs,
            outputs,
        }
    }

    /// Return the shapes of each graph input from the protobuf type info.
    ///
    /// Each inner `Vec<i64>` contains the dimension values.  Dynamic dims
    /// (symbolic or missing) are returned as -1.  Returns one entry per
    /// `graph.input` that has tensor type information.
    pub fn input_shapes(&self) -> Vec<Vec<i64>> {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut shapes = Vec::new();
        for inp in &graph.input {
            if let Some(type_proto) = &inp.r#type {
                if let Some(type_proto::Value::TensorType(tensor_type)) = &type_proto.value {
                    if let Some(shape) = &tensor_type.shape {
                        let dims: Vec<i64> = shape
                            .dim
                            .iter()
                            .map(|d| match &d.value {
                                Some(tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                                _ => -1,
                            })
                            .collect();
                        shapes.push(dims);
                    }
                }
            }
        }
        shapes
    }

    /// Number of weight-shaped initializers whose dtype is a non-FP32
    /// *floating-point* type (FP16, BF16, or Double).  Useful for the CLI
    /// to explain why `extract_weights` returned nothing on a model that
    /// visibly has data — most commonly an FP16-exported HuggingFace model.
    ///
    /// Only float-family dtypes are counted; rank-≥2 INT64 tensors are
    /// usually shape constants for `Reshape` / `Tile` / `Gather` and would
    /// otherwise show up as "non-FP32 weights" in the error message,
    /// confusing users.
    pub fn count_non_fp32_weight_initializers(&self) -> usize {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return 0,
        };
        let fp32 = tensor_proto::DataType::Float as i32;
        let fp16 = tensor_proto::DataType::Float16 as i32;
        let bf16 = tensor_proto::DataType::Bfloat16 as i32;
        let f64 = tensor_proto::DataType::Double as i32;
        graph
            .initializer
            .iter()
            .filter(|init| {
                init.dims.len() >= 2
                    && init.data_type != fp32
                    && (init.data_type == fp16 || init.data_type == bf16 || init.data_type == f64)
            })
            .count()
    }

    /// Number of initializers whose tensor data lives in an external file
    /// (`data_location == EXTERNAL`), rather than inline in the protobuf.
    ///
    /// quantize-rs reads only inline `raw_data` / `float_data`, so external-data
    /// tensors are skipped by [`extract_weights`](Self::extract_weights).  The
    /// CLI and Python layers use this to turn an otherwise-confusing "no weight
    /// tensors found" into a precise diagnostic: ONNX exports above ~2 GB
    /// (large LLMs in particular) commonly store weights in a sidecar
    /// `.onnx.data` file, which must be inlined before quantization.
    pub fn count_external_data_initializers(&self) -> usize {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return 0,
        };
        let external = tensor_proto::DataLocation::External as i32;
        graph
            .initializer
            .iter()
            .filter(|init| init.data_location == external)
            .count()
    }

    /// Extract the quantizable FP32 weight tensors from the model's initializers.
    ///
    /// Only **rank-≥2** tensors are returned.  Rank-0/1 initializers — biases,
    /// BatchNorm `scale`/`B`/`mean`/`var`, LayerNorm parameters, PRelu slopes —
    /// are not weights and must not be quantized: per-tensor INT8 on a BatchNorm
    /// `running_var` rounds near-zero variances to 0, and the `1/sqrt(var)` in
    /// BatchNorm then explodes the activations (this broke MobileNetV2 outright,
    /// cosine ≈ 0.10).  Genuine quantization targets are always rank ≥ 2 (Conv
    /// 4-D, MatMul/Gemm 2-D, embedding tables 2-D).
    ///
    /// QDQ scale scaffolding is also excluded: a `{base}_scale` FP32 initializer
    /// that has a sibling `{base}_quantized` is a DequantizeLinear scale (not a
    /// weight), as is any `_quantize_rs_`-prefixed initializer the save path
    /// synthesizes.  Without this, loading an already-quantized model and
    /// quantizing it again would quantize the scales and silently corrupt the
    /// dequantization.
    pub fn extract_weights(&self) -> Vec<WeightTensor> {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return Vec::new(),
        };

        // Initializer names, used to recognize QDQ scale scaffolding below.
        let init_names: std::collections::HashSet<&str> =
            graph.initializer.iter().map(|i| i.name.as_str()).collect();

        let mut weights = Vec::new();
        for initializer in &graph.initializer {
            // Only extract FP32 tensors — skip INT8, INT64, DOUBLE, etc.
            if initializer.data_type != tensor_proto::DataType::Float as i32 {
                continue;
            }

            // Skip rank-0/1 initializers: biases, BatchNorm parameters
            // (scale/B/mean/var), LayerNorm scale/bias, PRelu slopes, etc.  None
            // of these are the weight input of a Conv/MatMul/Gemm, and quantizing
            // them corrupts the model — BatchNorm `running_var` is the worst
            // case (near-zero variance → 0 under per-tensor INT8 → `1/sqrt(var)`
            // explodes).  A genuine quantizable weight is always rank ≥ 2.
            if initializer.dims.len() < 2 {
                continue;
            }

            // Skip quantize-rs scaffolding: internal synthesized initializers
            // and QDQ scale tensors (recognized by a sibling `_quantized`).
            if initializer.name.starts_with("_quantize_rs_") {
                continue;
            }
            if let Some(base) = initializer.name.strip_suffix("_scale") {
                if init_names.contains(format!("{base}_quantized").as_str()) {
                    continue;
                }
            }

            let name = initializer.name.clone();

            let shape: Vec<usize> = initializer
                .dims
                .iter()
                .map(|&d| d.max(0) as usize)
                .collect();

            let data = if !initializer.raw_data.is_empty() {
                if initializer.raw_data.len() % 4 != 0 {
                    // Misaligned raw_data — skip this initializer rather than panic
                    continue;
                }
                initializer
                    .raw_data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            } else {
                initializer.float_data.clone()
            };

            if !data.is_empty() {
                weights.push(WeightTensor { name, data, shape });
            }
        }

        weights
    }

    /// Total size of all weight tensors in bytes (float32).
    ///
    /// Prefer computing this from already-extracted weights when available:
    /// `weights.iter().map(|w| w.size_bytes()).sum()` avoids reparsing.
    pub fn total_size_bytes(&self) -> usize {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return 0,
        };
        graph
            .initializer
            .iter()
            .map(|init| {
                if !init.raw_data.is_empty() {
                    init.raw_data.len()
                } else {
                    init.float_data.len() * std::mem::size_of::<f32>()
                }
            })
            .sum()
    }
}

// ===========================================================================
// OnnxModel — quantized save (QDQ pattern, v0.3.0+)
// ===========================================================================

impl OnnxModel {
    /// Save a quantized model using the QDQ (DequantizeLinear) pattern.
    ///
    /// **Signature is identical to v0.2.0** — existing callers (CLI, calibration
    /// pipeline, examples) compile without changes.
    ///
    /// ### What changed internally
    ///
    /// v0.2.0 appended metadata to initializer names (e.g. `conv1.weight` →
    /// `conv1.weight__qINT8_s0.001_z-3_len9408`) without updating the nodes that
    /// reference them.  ONNX Runtime rejected these models on load.
    ///
    /// v0.3.0 inserts a `DequantizeLinear` node per weight.  The node's output
    /// carries the **original** name, so every downstream node is unchanged.
    /// Graph connectivity is preserved by construction, and the resulting model
    /// loads and runs in ONNX Runtime.
    ///
    /// ### INT4 storage note
    ///
    /// `DequantizeLinear` requires INT8 input in opsets &lt; 21.  By default,
    /// INT4-quantized values ([-8, 7]) are widened to INT8 bytes — 4×
    /// compression from FP32.  For true 8× compression, call
    /// [`save_quantized_with_options`](Self::save_quantized_with_options) with
    /// [`SaveOptions::with_native_int4`](graph_builder::SaveOptions::with_native_int4)`(true)`,
    /// which emits native `INT4` initializers and bumps the opset to 21.
    pub fn save_quantized(
        &mut self,
        quantized_data: &[graph_builder::QdqWeightInput],
        path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.save_quantized_with_options(quantized_data, path, SaveOptions::default())
    }

    /// Save a quantized model with explicit [`SaveOptions`] control.
    ///
    /// See [`save_quantized`](Self::save_quantized) for the transform details.
    /// Enabling [`SaveOptions::native_int4`] for INT4 weights bumps the
    /// required opset to 21 automatically.
    ///
    /// ### Fields not preserved on save
    ///
    /// quantize-rs models a subset of the ONNX schema, so re-encoding drops any
    /// section outside it: `ModelProto.functions` (local-function custom ops),
    /// `GraphProto.sparse_initializer`, `ModelProto.training_info`, and assorted
    /// `metadata_props`/`doc_string` on nodes and tensors.  When a *loaded* model
    /// carried `functions`, `sparse_initializer`, or `training_info`, this method
    /// prints a warning to stderr.  Models built from such sections (notably
    /// custom-op graphs) should be quantized with care — the dequantized weights
    /// are correct, but the saved graph will not contain those sections.
    pub fn save_quantized_with_options(
        &mut self,
        quantized_data: &[graph_builder::QdqWeightInput],
        path: impl AsRef<std::path::Path>,
        options: SaveOptions,
    ) -> Result<()> {
        let path = path.as_ref();
        use graph_builder::{apply_qdq_transform_with_options, ensure_opset_version};

        // Empty input list means "no quantization requested".  Treat this as
        // an error rather than silently bumping the opset and wiping any
        // existing `quantize_rs.bits.*` metadata.  Callers that genuinely
        // want to write the model unchanged should encode the protobuf and
        // write it themselves; the quantization save path is not the right
        // tool for that.
        if quantized_data.is_empty() {
            return Err(QuantizeError::GraphTransform {
                reason: "save_quantized_with_options called with an empty \
                         quantized_data slice; nothing to write.  Construct \
                         a non-empty Vec<QdqWeightInput> or save the proto \
                         directly via ModelProto::encode."
                    .to_string(),
            });
        }

        // --- 0c. Warn about wire-format sections this save will drop ---
        // The vendored ONNX schema is a subset, so re-encoding silently omits
        // anything outside it.  Most are inert, but local functions carry real
        // op semantics — a model that relies on them can be invalid after save.
        // Routed through the `log` facade (not `eprintln!`) so library and
        // Python consumers can suppress or redirect it; the CLI installs a
        // stderr logger so it still surfaces there.
        if self.dropped.any() {
            log::warn!(
                "the input model contains ONNX wire-format section(s) that quantize-rs \
                 does not preserve — {} will be ABSENT from '{}'.  Models that rely on local \
                 functions (custom ops) in particular may be invalid after quantization; verify \
                 the saved model before deploying.",
                self.dropped.describe(),
                path.display()
            );
        }

        // --- 1. Opset: ≥10 for per-tensor, ≥13 for per-channel, ≥21 for native INT4 ---
        let needs_per_channel = quantized_data.iter().any(|w| w.axis.is_some());
        let uses_native_int4 = options.native_int4 && quantized_data.iter().any(|w| w.bits == 4);
        let min_opset = if uses_native_int4 {
            21
        } else if needs_per_channel {
            13
        } else {
            10
        };
        ensure_opset_version(&mut self.proto, min_opset);

        // --- 2. Persist per-weight bits in model metadata ---
        // Drop any prior `quantize_rs.bits.*` entries before re-emitting so
        // repeated `save_quantized_with_options` calls on the same OnnxModel
        // do not accumulate duplicate metadata.
        self.proto
            .metadata_props
            .retain(|p| !p.key.starts_with("quantize_rs.bits."));
        for inp in quantized_data.iter() {
            self.proto.metadata_props.push(StringStringEntryProto {
                key: format!("quantize_rs.bits.{}", inp.original_name),
                value: inp.bits.to_string(),
            });
        }

        // --- 3. Apply QDQ transform to the graph ---
        let graph = self
            .proto
            .graph
            .as_mut()
            .ok_or_else(|| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: "Model has no graph".to_string(),
            })?;
        apply_qdq_transform_with_options(graph, quantized_data, options)?;

        // --- 4. Encode and write to disk atomically ---
        // Write to a sibling temp file and rename into place so a crash or
        // power loss mid-write does not leave a corrupted output at `path`.
        // Callers reloading the same path after a crash get either the old
        // file or the fully-written new file — never a torn write.
        let mut buf = Vec::new();
        self.proto
            .encode(&mut buf)
            .map_err(|e| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to encode ONNX model: {e}"),
            })?;

        // Unique temp path (pid + a process-local counter) so concurrent saves
        // to the same output path — across processes or threads — never share a
        // temp file and clobber each other.  The atomic rename below still makes
        // the final output appear all-at-once.
        let tmp_path = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut s = path.as_os_str().to_owned();
            s.push(format!(
                ".quantize-rs.{}.{}.tmp",
                std::process::id(),
                unique
            ));
            std::path::PathBuf::from(s)
        };

        // Encode → temp file → fsync.  If any step fails after the temp file
        // is created, remove the partial `.quantize-rs.tmp` before returning so
        // a failed save never leaves an orphan behind — mirroring the
        // rename-failure cleanup below.
        let write_tmp = || -> Result<()> {
            let mut file =
                std::fs::File::create(&tmp_path).map_err(|e| QuantizeError::ModelSave {
                    path: tmp_path.clone(),
                    reason: format!("Failed to create temp output file: {e}"),
                })?;

            file.write_all(&buf).map_err(|e| QuantizeError::ModelSave {
                path: tmp_path.clone(),
                reason: format!("Failed to write ONNX model: {e}"),
            })?;

            // Flush kernel buffers to disk before the rename so the new file
            // contents are durable.  Without this, the rename can succeed
            // before the data hits stable storage.
            file.sync_all().map_err(|e| QuantizeError::ModelSave {
                path: tmp_path.clone(),
                reason: format!("Failed to fsync ONNX model: {e}"),
            })?;

            Ok(())
        };
        if let Err(e) = write_tmp() {
            // Best-effort cleanup of the partial temp file.
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        std::fs::rename(&tmp_path, path).map_err(|e| {
            // Best-effort cleanup so we don't leave a stale .tmp file behind.
            // If even the cleanup fails (file locked by antivirus, EACCES,
            // etc.) we warn so the user knows there's a stray file to remove.
            if let Err(cleanup_err) = std::fs::remove_file(&tmp_path) {
                log::warn!(
                    "failed to clean up temporary file '{}' after \
                     rename failure: {} (please delete it manually)",
                    tmp_path.display(),
                    cleanup_err
                );
            }
            QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to rename temp file into place: {e}"),
            }
        })?;

        Ok(())
    }
}

// ===========================================================================
// OnnxModel — validation
// ===========================================================================

impl OnnxModel {
    /// Check that every node input in the graph resolves to a known tensor.
    ///
    /// A "known tensor" is one of:
    ///   - a declared graph input
    ///   - an initializer
    ///   - the output of a node appearing earlier in the node list
    ///
    /// This is the exact check ONNX Runtime performs on load.  It's the check
    /// that v0.2.0's `validate` command skipped, which is why the rename bug
    /// went undetected.  Integrate `report.summary()` into the CLI validate
    /// output alongside the existing structure / weight checks.
    pub fn validate_connectivity(&self) -> ConnectivityReport {
        match &self.proto.graph {
            Some(graph) => graph_builder::validate_graph_connectivity(graph),
            None => {
                use crate::onnx_proto::GraphProto;
                graph_builder::validate_graph_connectivity(&GraphProto::default())
            }
        }
    }
}

// ===========================================================================
// OnnxModel — quantized model introspection (v0.3.0 QDQ format)
// ===========================================================================

impl OnnxModel {
    /// Extract metadata about quantized weights from a QDQ-format model.
    ///
    /// Looks for initializer triples:
    ///   `{base}_quantized`, `{base}_scale`, `{base}_zp`
    ///
    /// Scale and zero-point are decoded in full — per-tensor yields a single
    /// element; per-channel yields one entry per channel.  Bit-width comes
    /// from `metadata_props` (written by `save_quantized`); defaults to 8 if
    /// the metadata entry is missing.
    ///
    /// Native INT4 zero-point tensors (`DataType::Int4`) are unpacked from
    /// their two-per-byte on-disk layout automatically.
    pub fn load_quantized_info(&self) -> Vec<QuantizedWeightInfo> {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut scale_map: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        let mut zp_map: std::collections::HashMap<String, Vec<i8>> =
            std::collections::HashMap::new();
        let mut quant_bases: Vec<String> = Vec::new();

        for init in &graph.initializer {
            let name = &init.name;

            if let Some(base) = name.strip_suffix("_scale") {
                scale_map.insert(base.to_string(), decode_scale_tensor(init));
            } else if let Some(base) = name.strip_suffix("_zp") {
                zp_map.insert(base.to_string(), decode_zero_point_tensor(init));
            } else if let Some(base) = name.strip_suffix("_quantized") {
                quant_bases.push(base.to_string());
            }
        }

        // Read bits from metadata_props (written by save_quantized)
        let mut bits_map: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
        for prop in &self.proto.metadata_props {
            if let Some(base) = prop.key.strip_prefix("quantize_rs.bits.") {
                if let Ok(bits) = prop.value.parse::<u8>() {
                    bits_map.insert(base.to_string(), bits);
                }
            }
        }

        quant_bases
            .iter()
            .map(|base| {
                let scales = scale_map.get(base).cloned().unwrap_or_else(|| vec![1.0]);
                let zero_points = zp_map.get(base).cloned().unwrap_or_else(|| vec![0]);
                let bits = bits_map.get(base).copied().unwrap_or(8);

                // Element count = product of dims on the _quantized tensor;
                // byte count = actual raw_data length (accounts for native INT4 packing).
                let quant_init = graph
                    .initializer
                    .iter()
                    .find(|i| i.name == format!("{}_quantized", base));
                let original_length = quant_init
                    .map(|i| i.dims.iter().product::<i64>() as usize)
                    .unwrap_or(0);
                let storage_bytes = quant_init.map(|i| i.raw_data.len()).unwrap_or(0);

                QuantizedWeightInfo {
                    name: base.clone(),
                    bits,
                    scales,
                    zero_points,
                    original_length,
                    storage_bytes,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers for load_quantized_info
// ---------------------------------------------------------------------------

/// Expected element count for a 1-D or scalar tensor: rank-0 → 1, rank-1 → dims[0].
fn expected_element_count(init: &crate::onnx_proto::TensorProto) -> usize {
    if init.dims.is_empty() {
        1
    } else {
        init.dims
            .iter()
            .copied()
            .filter(|&d| d > 0)
            .product::<i64>() as usize
    }
}

fn decode_scale_tensor(init: &crate::onnx_proto::TensorProto) -> Vec<f32> {
    let expected = expected_element_count(init).max(1);

    if !init.float_data.is_empty() {
        return init.float_data.clone();
    }

    if !init.raw_data.is_empty() && init.raw_data.len() >= 4 * expected {
        return init
            .raw_data
            .chunks_exact(4)
            .take(expected)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    }

    // Malformed or missing — fall back to a safe default so callers can still
    // report the weight exists without a division-by-zero risk.
    vec![1.0; expected]
}

fn decode_zero_point_tensor(init: &crate::onnx_proto::TensorProto) -> Vec<i8> {
    use crate::onnx_proto::tensor_proto::DataType;
    use crate::onnx_utils::quantization_nodes::unpack_int4_onnx;

    let expected = expected_element_count(init).max(1);

    // Native INT4: raw_data is packed two-per-byte, logical count in dims.
    if init.data_type == DataType::Int4 as i32 {
        return unpack_int4_onnx(&init.raw_data, expected);
    }

    // INT8 / widened INT4 / UINT8: raw_data is one byte per value.
    if !init.raw_data.is_empty() {
        return init
            .raw_data
            .iter()
            .take(expected)
            .map(|&b| b as i8)
            .collect();
    }

    // int32_data carries int-type scalars when raw_data is absent.
    if !init.int32_data.is_empty() {
        return init
            .int32_data
            .iter()
            .take(expected)
            .map(|&v| v as i8)
            .collect();
    }

    vec![0; expected]
}

// ===========================================================================
// WeightTensor (unchanged from v0.2.0)
// ===========================================================================

/// An FP32 weight tensor extracted from an ONNX model.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    /// Initializer name in the ONNX graph.
    pub name: String,
    /// FP32 weight values.
    pub data: Vec<f32>,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
}

impl WeightTensor {
    /// Size of this tensor in bytes (as FP32).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Total number of scalar elements.
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_scan_detects_local_functions() {
        // Hand-encoded protobuf: ModelProto field 25 (functions), wire type 2,
        // length 0 → one empty FunctionProto.  The tag (25<<3)|2 = 202 encodes
        // as the varint [0xCA, 0x01]; this pins the probe to the ONNX spec
        // field number, so a future typo can't silently disable the warning.
        let bytes = [0xCA, 0x01, 0x00];
        let d = DroppedSections::scan(&bytes);
        assert_eq!(d.functions, 1);
        assert!(d.any());
        assert!(d.describe().contains("local function"));
    }

    #[test]
    fn dropped_scan_detects_sparse_initializer() {
        // ModelProto.graph (field 7, wire 2) → GraphProto.sparse_initializer
        // (field 15, wire 2), each length 0.
        let bytes = [0x3A, 0x02, 0x7A, 0x00];
        let d = DroppedSections::scan(&bytes);
        assert_eq!(d.sparse_initializers, 1);
        assert!(d.any());
    }

    #[test]
    fn dropped_scan_clean_model_reports_nothing() {
        use prost::Message;
        let proto = ModelProto {
            ir_version: 7,
            ..Default::default()
        };
        let mut buf = Vec::new();
        proto.encode(&mut buf).unwrap();
        let d = DroppedSections::scan(&buf);
        assert!(!d.any(), "clean model should drop nothing: {d:?}");
    }
}
