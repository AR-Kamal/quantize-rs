//! Calibration methods for quantization range optimization.

use std::fmt;
use std::str::FromStr;

/// Strategy for choosing the quantization range from observed activations.
///
/// Marked `#[non_exhaustive]` so future calibration methods can be added
/// without a breaking change.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
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
    /// Renders the method in a form that round-trips through
    /// [`FromStr`].  `Percentile(p)` is rendered as `"percentile:{p}"` so
    /// the parameter survives the trip; the other variants render as their
    /// lowercase keyword.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationMethod::MinMax => write!(f, "minmax"),
            CalibrationMethod::Percentile(p) => write!(f, "percentile:{}", p),
            CalibrationMethod::Entropy => write!(f, "entropy"),
            CalibrationMethod::MSE => write!(f, "mse"),
        }
    }
}

impl FromStr for CalibrationMethod {
    type Err = crate::errors::QuantizeError;

    /// Parse a calibration method name.  Accepts:
    ///   - `"minmax"`, `"entropy"`, `"mse"` (case-insensitive)
    ///   - `"percentile"` (defaults to the 99.9th percentile)
    ///   - `"percentile:NN"` for an explicit percentile, e.g. `"percentile:95"`
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        if let Some(rest) = lower.strip_prefix("percentile:") {
            let p: f32 = rest
                .parse()
                .map_err(|_| crate::errors::QuantizeError::Config {
                    reason: format!(
                        "Invalid percentile '{}'; expected a number like 'percentile:99.9'",
                        rest
                    ),
                })?;
            if !(0.0..=100.0).contains(&p) {
                return Err(crate::errors::QuantizeError::Config {
                    reason: format!("Percentile must be in [0, 100], got {}", p),
                });
            }
            return Ok(CalibrationMethod::Percentile(p));
        }
        match lower.as_str() {
            "minmax" => Ok(CalibrationMethod::MinMax),
            "percentile" => Ok(CalibrationMethod::Percentile(99.9)),
            "entropy" => Ok(CalibrationMethod::Entropy),
            "mse" => Ok(CalibrationMethod::MSE),
            _ => Err(crate::errors::QuantizeError::Config {
                reason: format!(
                    "Unknown calibration method: '{}'. Valid: minmax, percentile, percentile:N, entropy, mse",
                    s
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_roundtrip() {
        // Display → FromStr → Display preserves the percentile value.
        let p1 = CalibrationMethod::Percentile(95.5);
        let s = format!("{}", p1);
        assert_eq!(s, "percentile:95.5");
        let p2: CalibrationMethod = s.parse().unwrap();
        assert!(matches!(p2, CalibrationMethod::Percentile(p) if (p - 95.5).abs() < 1e-6));
    }

    #[test]
    fn test_bare_percentile_default() {
        // "percentile" with no value falls back to 99.9 for back-compat.
        let m: CalibrationMethod = "percentile".parse().unwrap();
        assert!(matches!(m, CalibrationMethod::Percentile(p) if (p - 99.9).abs() < 1e-6));
    }

    #[test]
    fn test_invalid_percentile_rejected() {
        assert!("percentile:NaN_oops".parse::<CalibrationMethod>().is_err());
        assert!("percentile:-1".parse::<CalibrationMethod>().is_err());
        assert!("percentile:101".parse::<CalibrationMethod>().is_err());
    }

    #[test]
    fn test_all_keywords_roundtrip() {
        for method in [
            CalibrationMethod::MinMax,
            CalibrationMethod::Entropy,
            CalibrationMethod::MSE,
        ] {
            let s = format!("{}", method);
            let parsed: CalibrationMethod = s.parse().unwrap();
            assert_eq!(
                std::mem::discriminant(&method),
                std::mem::discriminant(&parsed)
            );
        }
    }
}
