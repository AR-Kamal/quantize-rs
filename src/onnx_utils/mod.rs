// src/onnx_utils/mod.rs
//! ONNX model utilities — loading, weight extraction, quantized save (QDQ),
//! graph connectivity validation, and quantized-model introspection.

pub mod graph_builder;
pub mod quantization_nodes;

use crate::errors::{QuantizeError, Result};
use protobuf::Message;
use std::fs;
use std::io::Read;

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
    proto: onnx::onnx::ModelProto,
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let graph = self.proto.get_graph();
        f.debug_struct("OnnxModel")
            .field("name", &graph.get_name())
            .field("num_nodes", &graph.get_node().len())
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

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to read ONNX file: {e}"),
            })?;

        let mut proto = onnx::onnx::ModelProto::new();
        proto
            .merge_from_bytes(&buffer)
            .map_err(|e| QuantizeError::ModelLoad {
                path: path.to_path_buf(),
                reason: format!("Failed to parse ONNX protobuf: {e}"),
            })?;

        Ok(Self { proto })
    }

    /// Return a summary of the model's structure.
    pub fn info(&self) -> ModelInfo {
        let graph = self.proto.get_graph();

        let inputs: Vec<String> = graph
            .get_input()
            .iter()
            .map(|i| i.get_name().to_string())
            .collect();

        let outputs: Vec<String> = graph
            .get_output()
            .iter()
            .map(|o| o.get_name().to_string())
            .collect();

        ModelInfo {
            name: graph.get_name().to_string(),
            version: self.proto.get_model_version(),
            num_nodes: graph.get_node().len(),
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
        let graph = self.proto.get_graph();
        let mut shapes = Vec::new();

        for inp in graph.get_input().iter() {
            if inp.has_field_type() {
                let type_proto = inp.get_field_type();
                if type_proto.has_tensor_type() {
                    let tensor_type = type_proto.get_tensor_type();
                    if tensor_type.has_shape() {
                        let shape_proto = tensor_type.get_shape();
                        let dims: Vec<i64> = shape_proto.get_dim().iter().map(|d| {
                            let v = d.get_dim_value();
                            if v > 0 { v } else { -1 }
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
        let mut weights = Vec::new();
        let graph = self.proto.get_graph();

        for initializer in graph.get_initializer() {
            let name = initializer.get_name().to_string();

            let shape: Vec<usize> = initializer.get_dims().iter()
                .map(|&d| d.max(0) as usize)
                .collect();

            let data = if initializer.has_raw_data() {
                let raw = initializer.get_raw_data();
                raw.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            } else {
                initializer.get_float_data().to_vec()
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
        let graph = self.proto.get_graph();
        graph.get_initializer().iter()
            .map(|init| {
                if init.has_raw_data() {
                    init.get_raw_data().len()
                } else {
                    std::mem::size_of_val(init.get_float_data())
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
            let mut prop = onnx::onnx::StringStringEntryProto::new();
            prop.set_key(format!("quantize_rs.bits.{}", inp.original_name));
            prop.set_value(inp.bits.to_string());
            self.proto.mut_metadata_props().push(prop);
        }

        // --- 3. Apply QDQ transform to the graph ---
        apply_qdq_transform(self.proto.mut_graph(), quantized_data)?;

        // --- 5. Write to disk ---
        let mut file = std::fs::File::create(path)
            .map_err(|e| QuantizeError::ModelSave {
                path: path.to_path_buf(),
                reason: format!("Failed to create output file: {e}"),
            })?;

        self.proto
            .write_to_writer(&mut file)
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
        graph_builder::validate_graph_connectivity(self.proto.get_graph())
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
        let graph = self.proto.get_graph();

        let mut scale_map: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut zp_map: std::collections::HashMap<String, i8> =
            std::collections::HashMap::new();
        let mut quant_bases: Vec<String> = Vec::new();

        for init in graph.get_initializer().iter() {
            let name = init.get_name();

            if let Some(base) = name.strip_suffix("_scale") {
                // Scale is stored in float_data (rank-0 scalar)
                let scale = if !init.get_float_data().is_empty() {
                    init.get_float_data()[0]
                } else {
                    // Fallback: try raw_data as little-endian f32
                    let raw = init.get_raw_data();
                    if raw.len() >= 4 {
                        f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
                    } else {
                        1.0
                    }
                };
                scale_map.insert(base.to_string(), scale);
            } else if let Some(base) = name.strip_suffix("_zp") {
                // Zero-point is a single raw byte
                let zp = if !init.get_raw_data().is_empty() {
                    init.get_raw_data()[0] as i8
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
        for prop in self.proto.get_metadata_props().iter() {
            if let Some(base) = prop.get_key().strip_prefix("quantize_rs.bits.") {
                if let Ok(bits) = prop.get_value().parse::<u8>() {
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
                let original_length = graph
                    .get_initializer()
                    .iter()
                    .find(|i| i.get_name() == format!("{}_quantized", base))
                    .map(|i| i.get_dims().iter().product::<i64>() as usize)
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