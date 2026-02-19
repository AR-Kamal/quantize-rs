//! YAML and TOML configuration file support.
//!
//! A configuration file can specify global quantization settings
//! (`bits`, `per_channel`), per-model overrides, and batch processing
//! parameters.

use crate::errors::{QuantizeError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level quantization configuration.
///
/// Can be loaded from a YAML or TOML file with [`Config::from_file`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default bit width (4 or 8). Defaults to 8.
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Default per-channel setting. Defaults to `false`.
    #[serde(default)]
    pub per_channel: bool,

    /// Layer names to exclude from quantization globally.
    #[serde(default)]
    pub excluded_layers: Vec<String>,

    /// Minimum number of elements a tensor must have to be quantized.
    /// Tensors smaller than this are kept in FP32. Defaults to 0 (no minimum).
    #[serde(default)]
    pub min_elements: usize,

    /// Per-model configuration overrides.
    #[serde(default)]
    pub models: Vec<ModelConfig>,

    /// Batch processing configuration.
    #[serde(default)]
    pub batch: Option<BatchConfig>,
}

/// Per-model quantization overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the input ONNX model.
    pub input: String,

    /// Path for the quantized output model.
    pub output: String,

    /// Override bit width for this model.
    #[serde(default)]
    pub bits: Option<u8>,

    /// Override per-channel setting for this model.
    #[serde(default)]
    pub per_channel: Option<bool>,

    /// Skip this model if the output file already exists.
    #[serde(default)]
    pub skip_existing: bool,

    /// Layer names to exclude from quantization for this model.
    /// Merged with (but does not replace) the global `excluded_layers`.
    #[serde(default)]
    pub excluded_layers: Vec<String>,

    /// Per-layer bit-width overrides for this model.
    /// Key = initializer name, value = 4 or 8.
    #[serde(default)]
    pub layer_bits: std::collections::HashMap<String, u8>,

    /// Override the global `min_elements` threshold for this model.
    #[serde(default)]
    pub min_elements: Option<usize>,
}

/// Batch processing configuration for quantizing multiple models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Glob pattern or directory for input models.
    pub input_dir: String,

    /// Output directory for quantized models.
    pub output_dir: String,

    /// Skip models whose output already exists.
    #[serde(default)]
    pub skip_existing: bool,

    /// Continue processing remaining models after a failure.
    #[serde(default)]
    pub continue_on_error: bool,
}

fn default_bits() -> u8 {
    8
}

impl Config {
    /// Load a config from a YAML or TOML file (auto-detected by extension).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Config`] on I/O, parse, or unsupported format errors.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| QuantizeError::Config { reason: "Config file has no extension".into() })?;

        let content = std::fs::read_to_string(path)
            .map_err(|e| QuantizeError::Config { reason: format!("Failed to read config file '{}': {e}", path.display()) })?;

        match extension {
            "yaml" | "yml" => Self::from_yaml(&content),
            "toml" => Self::from_toml(&content),
            _ => Err(QuantizeError::Config { reason: format!("Unsupported config format: {}", extension) }),
        }
    }

    /// Parse configuration from a YAML string.
    pub fn from_yaml(content: &str) -> Result<Self> {
        serde_yaml::from_str(content)
            .map_err(|e| QuantizeError::Config { reason: format!("Failed to parse YAML config: {e}") })
    }

    /// Parse configuration from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self> {
        toml::from_str(content)
            .map_err(|e| QuantizeError::Config { reason: format!("Failed to parse TOML config: {e}") })
    }

    /// Validate the configuration (bits values, non-empty paths).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Config`] if any field is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.bits != 4 && self.bits != 8 {
            return Err(QuantizeError::Config { reason: format!("Invalid bits value: {}. Must be 4 or 8", self.bits) });
        }

        for (idx, model) in self.models.iter().enumerate() {
            if model.input.is_empty() {
                return Err(QuantizeError::Config { reason: format!("Model {}: input path is empty", idx) });
            }
            if model.output.is_empty() {
                return Err(QuantizeError::Config { reason: format!("Model {}: output path is empty", idx) });
            }
            if let Some(bits) = model.bits {
                if bits != 4 && bits != 8 {
                    return Err(QuantizeError::Config { reason: format!("Model {}: invalid bits value: {}", idx, bits) });
                }
            }
            for (layer, &bits) in &model.layer_bits {
                if bits != 4 && bits != 8 {
                    return Err(QuantizeError::Config { reason: format!("Model {}: invalid bits {} for layer '{}'", idx, bits, layer) });
                }
            }
        }

        if let Some(batch) = &self.batch {
            if batch.input_dir.is_empty() {
                return Err(QuantizeError::Config { reason: "Batch input_dir is empty".into() });
            }
            if batch.output_dir.is_empty() {
                return Err(QuantizeError::Config { reason: "Batch output_dir is empty".into() });
            }
        }

        Ok(())
    }

    /// Effective bit width for a model (model override or global default).
    pub fn get_bits(&self, model: &ModelConfig) -> u8 {
        model.bits.unwrap_or(self.bits)
    }

    /// Effective per-channel setting for a model (model override or global default).
    pub fn get_per_channel(&self, model: &ModelConfig) -> bool {
        model.per_channel.unwrap_or(self.per_channel)
    }

    /// Effective excluded-layers list: global list merged with model-level list.
    pub fn get_excluded_layers(&self, model: &ModelConfig) -> Vec<String> {
        let mut layers = self.excluded_layers.clone();
        for l in &model.excluded_layers {
            if !layers.contains(l) {
                layers.push(l.clone());
            }
        }
        layers
    }

    /// Effective min-elements threshold for a model.
    pub fn get_min_elements(&self, model: &ModelConfig) -> usize {
        model.min_elements.unwrap_or(self.min_elements)
    }

    /// Effective per-layer bit-width overrides for a model.
    ///
    /// Layer names are model-specific so there is no global map to merge;
    /// this simply returns the model's own `layer_bits` map.
    pub fn get_layer_bits(&self, model: &ModelConfig) -> std::collections::HashMap<String, u8> {
        model.layer_bits.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yaml_config() {
        let yaml = r#"
bits: 8
per_channel: true

models:
  - input: model1.onnx
    output: model1_int8.onnx
  
  - input: model2.onnx
    output: model2_int8.onnx
    per_channel: false

batch:
  input_dir: "models/*.onnx"
  output_dir: quantized/
  skip_existing: true
"#;

        let config = Config::from_yaml(yaml).unwrap();
        assert_eq!(config.bits, 8);
        assert!(config.per_channel);
        assert_eq!(config.models.len(), 2);
        assert!(config.batch.is_some());
    }

    #[test]
    fn test_toml_config() {
        let toml = r#"
bits = 8
per_channel = true

[[models]]
input = "model1.onnx"
output = "model1_int8.onnx"

[[models]]
input = "model2.onnx"
output = "model2_int8.onnx"
per_channel = false

[batch]
input_dir = "models/*.onnx"
output_dir = "quantized/"
skip_existing = true
"#;

        let config = Config::from_toml(toml).unwrap();
        assert_eq!(config.bits, 8);
        assert!(config.per_channel);
        assert_eq!(config.models.len(), 2);
        assert!(config.batch.is_some());
    }
}