//! # Quantify
//!
//! A simple neural network quantization toolkit.
//!
//! Converts ONNX models to INT8/INT4 with minimal accuracy loss.

pub mod quantization;
pub mod onnx_utils;

pub use quantization::{Quantizer, QuantConfig, QuantMode};
pub use onnx_utils::{OnnxModel, ModelInfo};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}