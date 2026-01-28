pub mod errors;
pub mod onnx_utils;
pub mod quantization;
pub mod config;
pub mod calibration;  

pub use onnx_utils::{ModelInfo, OnnxModel, WeightTensor, QuantizedWeightInfo};
pub use quantization::{Quantizer, QuantConfig, QuantMode, QuantParams, pack_int4, unpack_int4};
pub use config::Config;
pub use calibration::{CalibrationDataset, stats::ActivationStats, inference::ActivationEstimator,};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
