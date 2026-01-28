//src/onnx_utils/mod.rs
//! ONNX model utilities
use anyhow::{Context, Result};
use protobuf::Message;
use std::fs;
use std::io::Read;

/// ONNX model wrapper
pub struct OnnxModel {
    proto: onnx::onnx::ModelProto,
}

/// Model information
#[derive(Debug)]
pub struct ModelInfo {
    pub name: String,
    pub version: i64,
    pub num_nodes: usize,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Information about a quantized weight tensor
#[derive(Debug, Clone)]
pub struct QuantizedWeightInfo {
    pub name: String,
    pub bits: u8,
    pub scale: f32,
    pub zero_point: i8,
    pub original_length: usize,
}

impl OnnxModel {
    /// Load ONNX model from file
    /// Load ONNX model from file
    pub fn load(path: &str) -> Result<Self> {
        // Read file bytes
        let mut file =
            fs::File::open(path).with_context(|| format!("Failed to open ONNX file: {}", path))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .context("Failed to read ONNX file")?;

        // Parse using protobuf - create instance first, then merge
        let mut proto = onnx::onnx::ModelProto::new();
        proto
            .merge_from_bytes(&buffer)
            .context("Failed to parse ONNX protobuf")?;

        Ok(Self { proto })
    }

    /// Get model information  
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

    /// Extract all weights from the model
    /// Weights are stored in graph.initializer
    pub fn extract_weights(&self) -> Vec<WeightTensor> {
        let mut weights = Vec::new();
        let graph = self.proto.get_graph();

        // Iterate through all initializers (this is where weights are stored!)
        for initializer in graph.get_initializer() {
            let name = initializer.get_name().to_string();

            // Get shape
            let shape: Vec<usize> = initializer.get_dims().iter().map(|&d| d as usize).collect();

            // Extract float32 data
            // ONNX stores data in different formats depending on data_type
            let data = if initializer.has_raw_data() {
                // Raw data format (more efficient for large tensors)
                let raw = initializer.get_raw_data();
                let float_data: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                float_data
            } else {
                // Float array format
                initializer.get_float_data().to_vec()
            };

            if !data.is_empty() {
                weights.push(WeightTensor { name, data, shape });
            }
        }

        weights
    }

    /// Get total model size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.extract_weights()
            .iter()
            .map(|w| w.data.len() * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn save_quantized(
        &mut self, 
        quantized_data: &[(String, Vec<i8>, f32, i8, u8)], 
        path: &str
    ) -> Result<()> {
        let graph = self.proto.mut_graph();
        
        for (name, quant_data, scale, zero_point, bits) in quantized_data {
            if let Some(init) = graph.mut_initializer().iter_mut().find(|i| {
                i.get_name() == name
            }) {
                // Clear existing data
                init.clear_float_data();
                init.clear_raw_data();
                
                // Store quantized data based on bits
                let raw_bytes: Vec<u8> = if *bits == 4 {
                    // INT4: Pack 2 values per byte for 8x compression
                    use crate::quantization::pack_int4;
                    pack_int4(quant_data)
                } else {
                    // INT8: Store as-is for 4x compression
                    quant_data.iter().map(|&v| v as u8).collect()
                };
                
                init.set_raw_data(raw_bytes);
                init.set_data_type(onnx::onnx::TensorProto_DataType::INT8);
                
                // Encode metadata in name
                // Format: original_name__qINT{bits}_s{scale}_z{zero_point}_len{original_length}
                let current_name = init.get_name();
                if !current_name.contains("__qINT") {
                    let original_len = quant_data.len();
                    let metadata = format!(
                        "__qINT{}_s{:.8}_z{}_len{}", 
                        bits, 
                        scale, 
                        zero_point,
                        original_len
                    );
                    init.set_name(format!("{}{}", current_name, metadata));
                }
            }
        }

        let mut file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create output file: {}", path))?;
        
        self.proto.write_to_writer(&mut file)
            .context("Failed to write ONNX model")?;

        Ok(())
    }

    /// Load a quantized model and decode the quantization metadata
    pub fn load_quantized_info(&self) -> Vec<QuantizedWeightInfo> {
        let mut infos = Vec::new();
        let graph = self.proto.get_graph();
        
        for init in graph.get_initializer() {
            let name = init.get_name();
            
            // Check if this is a quantized tensor
            if name.contains("__qINT") {
                // Parse metadata from name
                if let Some(metadata_start) = name.find("__qINT") {
                    let metadata = &name[metadata_start..];
                    
                    // Extract bits
                    let bits = if metadata.contains("__qINT4") {
                        4
                    } else if metadata.contains("__qINT8") {
                        8
                    } else {
                        continue;
                    };
                    
                    // Extract scale (format: _s{scale}_)
                    let scale = if let Some(s_pos) = metadata.find("_s") {
                        let scale_str = &metadata[s_pos + 2..];
                        if let Some(end) = scale_str.find('_') {
                            scale_str[..end].parse::<f32>().unwrap_or(1.0)
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    
                    // Extract zero_point (format: _z{zero_point}_)
                    let zero_point = if let Some(z_pos) = metadata.find("_z") {
                        let zp_str = &metadata[z_pos + 2..];
                        if let Some(end) = zp_str.find('_') {
                            zp_str[..end].parse::<i8>().unwrap_or(0)
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    
                    // Extract original length
                    let original_len = if let Some(len_pos) = metadata.find("_len") {
                        let len_str = &metadata[len_pos + 4..];
                        len_str.parse::<usize>().unwrap_or(0)
                    } else {
                        0
                    };
                    
                    // Get original name (before metadata)
                    let original_name = name[..metadata_start].to_string();
                    
                    infos.push(QuantizedWeightInfo {
                        name: original_name,
                        bits,
                        scale,
                        zero_point,
                        original_length: original_len,
                    });
                }
            }
        }
        
        infos
    }
}

/// A weight tensor extracted from the model
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl WeightTensor {
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}
