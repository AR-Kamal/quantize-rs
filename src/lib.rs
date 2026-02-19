//! Neural network quantization toolkit for ONNX models.
//!
//! `quantize-rs` converts FP32 ONNX model weights to INT8 or INT4,
//! reducing model size by 4--8x with minimal accuracy loss. It supports
//! per-tensor and per-channel quantization, calibration-based range
//! optimization, and writes ONNX-Runtime-compatible QDQ models.
//!
//! # Modules
//!
//! - [`quantization`] -- core quantization logic (INT8/INT4, per-channel, packing)
//! - [`onnx_utils`] -- ONNX model loading, weight extraction, QDQ save, validation
//! - [`calibration`] -- (feature `calibration`) calibration datasets, activation-based inference, range methods
//! - [`config`] -- YAML/TOML configuration file support
//! - [`errors`] -- typed error enum ([`QuantizeError`]) for all public API functions
//!
//! # Feature flags
//!
//! - **`calibration`** *(default)* -- enables activation-based calibration (adds `tract-onnx`, `ndarray`)
//! - **`python`** -- enables PyO3 bindings (`quantize_rs` Python module)

pub mod errors;
pub mod onnx_proto;
pub mod onnx_utils;
pub mod quantization;
pub mod config;
pub mod calibration;

pub use errors::QuantizeError;
pub use onnx_utils::{ModelInfo, OnnxModel, WeightTensor, QuantizedWeightInfo, ConnectivityReport};
pub use onnx_utils::graph_builder::QdqWeightInput;
pub use quantization::{Quantizer, QuantConfig, QuantParams, pack_int4, unpack_int4};
pub use config::Config;
pub use calibration::{CalibrationDataset, stats::ActivationStats};
#[cfg(feature = "calibration")]
pub use calibration::inference::ActivationEstimator;

/// Library version string, read from `Cargo.toml` at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

#[cfg(feature = "python")]
mod python;