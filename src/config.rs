//! Configuration file support

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default quantization bits
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Default per-channel setting
    #[serde(default)]
    pub per_channel: bool,

    /// Individual model configurations
    #[serde(default)]
    pub models: Vec<ModelConfig>,

    /// Batch configuration
    #[serde(default)]
    pub batch: Option<BatchConfig>,
}

/// Configuration for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input model path
    pub input: String,

    /// Output model path
    pub output: String,

    /// Override global bits setting
    #[serde(default)]
    pub bits: Option<u8>,

    /// Override global per_channel setting
    #[serde(default)]
    pub per_channel: Option<bool>,

    /// Skip if output already exists
    #[serde(default)]
    pub skip_existing: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Input directory or glob pattern
    pub input_dir: String,

    /// Output directory
    pub output_dir: String,

    /// Skip existing files
    #[serde(default)]
    pub skip_existing: bool,

    /// Continue on errors
    #[serde(default)]
    pub continue_on_error: bool,
}

fn default_bits() -> u8 {
    8
}

impl Config {
    /// Load configuration from file (auto-detect format)
    pub fn from_file(path: &str) -> Result<Self> {
        let path = Path::new(path);
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .context("Config file has no extension")?;

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        match extension {
            "yaml" | "yml" => Self::from_yaml(&content),
            "toml" => Self::from_toml(&content),
            _ => anyhow::bail!("Unsupported config format: {}", extension),
        }
    }

    /// Parse YAML configuration
    pub fn from_yaml(content: &str) -> Result<Self> {
        serde_yaml::from_str(content)
            .context("Failed to parse YAML config")
    }

    /// Parse TOML configuration
    pub fn from_toml(content: &str) -> Result<Self> {
        toml::from_str(content)
            .context("Failed to parse TOML config")
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Check bits
        if self.bits != 4 && self.bits != 8 {
            anyhow::bail!("Invalid bits value: {}. Must be 4 or 8", self.bits);
        }

        // Check models
        for (idx, model) in self.models.iter().enumerate() {
            if model.input.is_empty() {
                anyhow::bail!("Model {}: input path is empty", idx);
            }
            if model.output.is_empty() {
                anyhow::bail!("Model {}: output path is empty", idx);
            }
            if let Some(bits) = model.bits {
                if bits != 4 && bits != 8 {
                    anyhow::bail!("Model {}: invalid bits value: {}", idx, bits);
                }
            }
        }

        // Check batch config
        if let Some(batch) = &self.batch {
            if batch.input_dir.is_empty() {
                anyhow::bail!("Batch input_dir is empty");
            }
            if batch.output_dir.is_empty() {
                anyhow::bail!("Batch output_dir is empty");
            }
        }

        Ok(())
    }

    /// Get effective bits for a model (model override or global default)
    pub fn get_bits(&self, model: &ModelConfig) -> u8 {
        model.bits.unwrap_or(self.bits)
    }

    /// Get effective per_channel for a model
    pub fn get_per_channel(&self, model: &ModelConfig) -> bool {
        model.per_channel.unwrap_or(self.per_channel)
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