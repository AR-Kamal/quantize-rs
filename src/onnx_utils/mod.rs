// src/onnx_utils/mod.rs
//! ONNX model utilities — loading, weight extraction, quantized save (QDQ),
//! graph connectivity validation, and quantized-model introspection.

pub mod graph_builder;
pub mod quantization_nodes;

use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{
    ModelProto, StringStringEntryProto,
    tensor_proto, tensor_shape_proto, type_proto,
};
use prost::Message;
use std::fs;
use std::io::{Read, Write};

// Re-export so callers don't have to reach into submodules
pub use graph_builder::ConnectivityReport;

// ===========================================================================
// Core types
// ===========================================================================

/// An ONNX model loaded from a protobuf file.
///
/// Provides methods for inspecting, extracting weights, saving quantized
/// models, and validating graph connectivity.
pub struct OnnxModel {
    proto: ModelProto,
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.proto.graph.as_ref().map(|g| g.name.as_str()).unwrap_or("");
        let num_nodes = self.proto.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);
        f.debug_struct("OnnxModel")
            .field("name", &name)
            .field("num_nodes", &num_nodes)
            .finish()
    }
}

/// Summary of an ONNX model's structure.
#[derive(Debug)]
pub struct ModelInfo {
    /// Graph name from the protobuf.
    pub name: String,
    /// Model version from the protobuf.
    pub version: i64,
    /// Number of computation nodes in the graph.
    pub num_nodes: usize,
    /// Names of the graph inputs.
    pub inputs: Vec<String>,
    /// Names of the graph outputs.
    pub outputs: Vec<String>,
}

/// Metadata about a quantized weight recovered from a QDQ-format model.
#[derive(Debug, Clone)]
pub struct QuantizedWeightInfo {
    /// Original weight name (without `_quantized` suffix).
    pub name: String,
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Quantization scale factor.
    pub scale: f32,
    /// Quantization zero point.
    pub zero_point: i8,
    /// Number of elements in the quantized tensor.
    pub original_length: usize,
}

// ===========================================================================
// OnnxModel — load / inspect
// ===========================================================================

impl OnnxModel {
    /// Load an ONNX model from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::ModelLoad`] if the file cannot be opened,
    /// is too large (>10 GB), or contains invalid protobuf data.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            fs::File::open(path).map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to open ONNX file: {e}"),
            })?;

        const MAX_MODEL_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10 GB
        let file_size = file.metadata()
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to read metadata: {e}"),
            })?
            .len();
        if file_size > MAX_MODEL_SIZE {
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

        let proto = ModelProto::decode(&buffer[..])
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to parse ONNX protobuf: {e}"),
            })?;

        Ok(Self { proto })
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

        ModelInfo {
            name: graph.map(|g| g.name.clone()).unwrap_or_default(),
            version: self.proto.model_version,
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
                        let dims: Vec<i64> = shape.dim.iter().map(|d| {
                            match &d.value {
                                Some(tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                                _ => -1,
                            }
                        }).collect();
                        shapes.push(dims);
                    }
                }
            }
        }
        shapes
    }

    /// Extract all FP32 weight tensors from the model's initializers.
    pub fn extract_weights(&self) -> Vec<WeightTensor> {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut weights = Vec::new();
        for initializer in &graph.initializer {
            // Only extract FP32 tensors — skip INT8, INT64, DOUBLE, etc.
            if initializer.data_type != tensor_proto::DataType::Float as i32 {
                continue;
            }

            let name = initializer.name.clone();

            let shape: Vec<usize> = initializer.dims.iter()
                .map(|&d| d.max(0) as usize)
                .collect();

            let data = if !initializer.raw_data.is_empty() {
                initializer.raw_data.chunks_exact(4)
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
        graph.initializer.iter()
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
    /// `DequantizeLinear` requires INT8 input (opset < 21).  INT4-quantized values
    /// ([-8, 7]) are stored as INT8 bytes.  Quantization *accuracy* is still
    /// INT4-level; only the on-disk size is 4× instead of the 8× that bit-packing
    /// would give.  True INT4 packing is a v0.4.0 target.
    pub fn save_quantized(
        &mut self,
        quantized_data: &[graph_builder::QdqWeightInput],
        path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let path = path.as_ref();
        use graph_builder::{apply_qdq_transform, ensure_opset_version};

        // --- 1. Opset ≥ 13 (DequantizeLinear per-channel needs it) ---
        ensure_opset_version(&mut self.proto, 13);

        // --- 2. Persist per-weight bits in model metadata ---
        for inp in quantized_data.iter() {
            self.proto.metadata_props.push(StringStringEntryProto {
                key:   format!("quantize_rs.bits.{}", inp.original_name),
                value: inp.bits.to_string(),
            });
        }

        // --- 3. Apply QDQ transform to the graph ---
        let graph = self.proto.graph.as_mut().ok_or_else(|| QuantizeError::ModelSave {
            path: path.to_path_buf(),
            reason: "Model has no graph".to_string(),
        })?;
        apply_qdq_transform(graph, quantized_data)?;

        // --- 4. Encode and write to disk ---
        let mut buf = Vec::new();
        self.proto.encode(&mut buf)
            .map_err(|e| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to encode ONNX model: {e}"),
            })?;

        let mut file = std::fs::File::create(path)
            .map_err(|e| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to create output file: {e}"),
            })?;

        file.write_all(&buf)
            .map_err(|e| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to write ONNX model: {e}"),
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
    /// Scale and zero-point values are read directly from the tensors.
    /// Bit-width comes from `metadata_props` (written by `save_quantized`);
    /// defaults to 8 if the metadata entry is missing.
    pub fn load_quantized_info(&self) -> Vec<QuantizedWeightInfo> {
        let graph = match &self.proto.graph {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut scale_map: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut zp_map: std::collections::HashMap<String, i8> =
            std::collections::HashMap::new();
        let mut quant_bases: Vec<String> = Vec::new();

        for init in &graph.initializer {
            let name = &init.name;

            if let Some(base) = name.strip_suffix("_scale") {
                // Scale is stored in float_data (rank-0 scalar)
                let scale = if !init.float_data.is_empty() {
                    init.float_data[0]
                } else if init.raw_data.len() >= 4 {
                    // Fallback: try raw_data as little-endian f32
                    f32::from_le_bytes([
                        init.raw_data[0], init.raw_data[1],
                        init.raw_data[2], init.raw_data[3],
                    ])
                } else {
                    1.0
                };
                scale_map.insert(base.to_string(), scale);
            } else if let Some(base) = name.strip_suffix("_zp") {
                // Zero-point is a single raw byte
                let zp = if !init.raw_data.is_empty() {
                    init.raw_data[0] as i8
                } else {
                    0
                };
                zp_map.insert(base.to_string(), zp);
            } else if let Some(base) = name.strip_suffix("_quantized") {
                quant_bases.push(base.to_string());
            }
        }

        // Read bits from metadata_props (written by save_quantized)
        let mut bits_map: std::collections::HashMap<String, u8> =
            std::collections::HashMap::new();
        for prop in &self.proto.metadata_props {
            if let Some(base) = prop.key.strip_prefix("quantize_rs.bits.") {
                if let Ok(bits) = prop.value.parse::<u8>() {
                    bits_map.insert(base.to_string(), bits);
                }
            }
        }

        // Assemble QuantizedWeightInfo from the three maps
        quant_bases
            .iter()
            .map(|base| {
                let scale = scale_map.get(base).copied().unwrap_or(1.0);
                let zp = zp_map.get(base).copied().unwrap_or(0);
                let bits = bits_map.get(base).copied().unwrap_or(8);

                // Element count = product of dims on the _quantized tensor
                let original_length = graph.initializer.iter()
                    .find(|i| i.name == format!("{}_quantized", base))
                    .map(|i| i.dims.iter().product::<i64>() as usize)
                    .unwrap_or(0);

                QuantizedWeightInfo {
                    name: base.clone(),
                    bits,
                    scale,
                    zero_point: zp,
                    original_length,
                }
            })
            .collect()
    }
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
