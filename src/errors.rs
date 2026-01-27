//! Error handling with helpful messages

use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuantifyError {
    #[error("Failed to load ONNX model '{path}': {source}")]
    ModelLoadError { path: String, source: anyhow::Error },

    #[error("Model has no weights to quantize. This might not be a trained model.")]
    NoWeightsFound,

    #[error("Unsupported quantization: {bits}-bit. Only INT8 (8-bit) is currently supported.")]
    UnsupportedBits { bits: u8 },

    #[error("File not found: '{path}'. Please check the path and try again.")]
    FileNotFound { path: String },

    #[error("Invalid ONNX file: {reason}")]
    InvalidOnnx { reason: String },

    #[error("Failed to save quantized model to '{path}': {source}")]
    SaveError { path: String, source: anyhow::Error },
}
