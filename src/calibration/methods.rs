// src/calibration/methods.rs
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub enum CalibrationMethod {
    #[default]
    MinMax,

    Percentile(f32),

    Entropy,

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
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minmax" => Ok(CalibrationMethod::MinMax),
            "percentile" => Ok(CalibrationMethod::Percentile(99.9)),
            "entropy" => Ok(CalibrationMethod::Entropy),
            "mse" => Ok(CalibrationMethod::MSE),
            _ => Err(anyhow::anyhow!("Unknown calibration method: '{}'. Valid methods: minmax, percentile, entropy, mse", s)),
        }
    }
}
