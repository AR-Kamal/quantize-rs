//! # quantize-rs
//!
//! A simple neural network quantization toolkit.
//!
//! Converts ONNX models to INT8/INT4 with minimal accuracy loss.

pub mod errors;
pub mod onnx_utils;
pub mod quantization;

pub use onnx_utils::{ModelInfo, OnnxModel, WeightTensor};
pub use quantization::{QuantConfig, QuantMode, QuantParams, Quantizer};

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
