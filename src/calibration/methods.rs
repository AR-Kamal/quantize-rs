//! Calibration methods for quantization range optimization.

use std::fmt;
use std::str::FromStr;

/// Strategy for choosing the quantization range from observed activations.
#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub enum CalibrationMethod {
    /// Use the full observed min/max range (default).
    #[default]
    MinMax,

    /// Clip outliers at the given percentile (e.g. 99.9).
    Percentile(f32),

    /// Minimize KL divergence between the original and quantized distributions.
    Entropy,

    /// Minimize mean squared error between original and dequantized values.
    MSE,
}


impl fmt::Display for CalibrationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationMethod::MinMax => write!(f, "MinMax"),
            CalibrationMethod::Percentile(_p) => write!(f, "Percentile"),
            CalibrationMethod::Entropy => write!(f, "Entropy"),
            CalibrationMethod::MSE => write!(f, "MSE"),
        }
    }
}

impl FromStr for CalibrationMethod {
    type Err = crate::errors::QuantizeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minmax" => Ok(CalibrationMethod::MinMax),
            "percentile" => Ok(CalibrationMethod::Percentile(99.9)),
            "entropy" => Ok(CalibrationMethod::Entropy),
            "mse" => Ok(CalibrationMethod::MSE),
            _ => Err(crate::errors::QuantizeError::Config {
                reason: format!("Unknown calibration method: '{}'. Valid methods: minmax, percentile, entropy, mse", s),
            }),
        }
    }
}
