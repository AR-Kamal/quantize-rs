// src/calibration/methods.rs
#[derive(Debug, Clone, Copy)]
pub enum CalibrationMethod {
    MinMax,
    
    Percentile(f32),

    Entropy,
    
    MSE,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::MinMax
    }
}

impl CalibrationMethod {
    pub fn name(&self) -> &str {
        match self {
            CalibrationMethod::MinMax => "MinMax",
            CalibrationMethod::Percentile(_p) => "Percentile",
            CalibrationMethod::Entropy => "Entropy",
            CalibrationMethod::MSE => "MSE",
        }
    }
}

// TODO: Implement optimization functions in Phase 5
// pub fn optimize_kl_divergence(stats: &ActivationStats) -> (f32, f32) { ... }
// pub fn optimize_mse(stats: &ActivationStats) -> (f32, f32) { ... }