//! Typed error handling for the quantize-rs library.
//!
//! All public API functions return [`Result<T>`](type@Result), which uses
//! [`QuantizeError`] as the error type. The CLI binary converts these into
//! `anyhow::Error` automatically via the blanket `From<E: std::error::Error>`
//! impl, so callers that prefer `anyhow` can use `?` without `.map_err()`.

use std::fmt;
use std::path::PathBuf;

/// Result type alias used throughout the quantize-rs public API.
pub type Result<T> = std::result::Result<T, QuantizeError>;

/// Errors produced by the quantize-rs library.
///
/// Each variant covers a distinct failure category. The `reason` field
/// carries a human-readable explanation suitable for display.
#[derive(Debug)]
pub enum QuantizeError {
    /// Empty tensor, shape mismatch, per-channel on a scalar, etc.
    InvalidTensor {
        /// What went wrong.
        reason: String,
    },

    /// Unsupported quantization configuration (e.g. bits != 4 or 8).
    UnsupportedConfig {
        /// What went wrong.
        reason: String,
    },

    /// Failed to load an ONNX model from disk.
    ModelLoad {
        /// Path that was being loaded.
        path: PathBuf,
        /// What went wrong.
        reason: String,
    },

    /// Failed to save a quantized ONNX model to disk.
    ModelSave {
        /// Path that was being written.
        path: PathBuf,
        /// What went wrong.
        reason: String,
    },

    /// Error during QDQ graph transformation (weight not found, size mismatch, etc.).
    GraphTransform {
        /// What went wrong.
        reason: String,
    },

    /// Error during calibration (invalid dataset, inference failure, etc.).
    Calibration {
        /// What went wrong.
        reason: String,
    },

    /// Configuration file parsing or validation error.
    Config {
        /// What went wrong.
        reason: String,
    },

    /// Catch-all for rare edge cases that don't fit other variants.
    Other(String),
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizeError::InvalidTensor { reason } => {
                write!(f, "invalid tensor: {reason}")
            }
            QuantizeError::UnsupportedConfig { reason } => {
                write!(f, "unsupported config: {reason}")
            }
            QuantizeError::ModelLoad { path, reason } => {
                write!(f, "failed to load model '{}': {reason}", path.display())
            }
            QuantizeError::ModelSave { path, reason } => {
                write!(f, "failed to save model '{}': {reason}", path.display())
            }
            QuantizeError::GraphTransform { reason } => {
                write!(f, "graph transform error: {reason}")
            }
            QuantizeError::Calibration { reason } => {
                write!(f, "calibration error: {reason}")
            }
            QuantizeError::Config { reason } => {
                write!(f, "config error: {reason}")
            }
            QuantizeError::Other(msg) => {
                write!(f, "{msg}")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}
