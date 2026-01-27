//src/onnx_utils/mod.rs
//! ONNX model utilities
use crate::quantization::QuantParams;
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
        quantized_weights: &[(String, Vec<i8>, QuantParams)],
        output_path: &str,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Get mutable graph
        let graph = self.proto.mut_graph();

        // Replace each initializer with quantized version
        for (name, quant_data, _params) in quantized_weights {
            // Find the initializer with this name
            for initializer in graph.mut_initializer().iter_mut() {
                if initializer.get_name() == name {
                    // Clear old float data
                    initializer.clear_float_data();
                    initializer.clear_raw_data();

                    // Set data type to INT8 using proper enum
                    initializer.set_data_type(onnx::onnx::TensorProto_DataType::INT8);

                    // Convert Vec<i8> to bytes
                    let bytes: Vec<u8> = quant_data.iter().map(|&v| v as u8).collect();

                    // Set raw data
                    initializer.set_raw_data(bytes);

                    // TODO: Store scale and zero-point as separate tensors
                    // (this is simplified - production would add these as initializers)

                    break;
                }
            }
        }

        // Serialize to bytes
        let bytes = self
            .proto
            .write_to_bytes()
            .context("Failed to serialize ONNX model")?;

        // Write to file
        let mut file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;

        file.write_all(&bytes)
            .context("Failed to write ONNX model to file")?;

        Ok(())
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
