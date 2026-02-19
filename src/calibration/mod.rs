//! Calibration datasets and activation-based range estimation.
//!
//! - [`CalibrationDataset`] — load or generate calibration samples
//! - [`methods::CalibrationMethod`] — range optimization strategies
//! - [`stats::ActivationStats`] — incremental min/max/histogram tracker
//! - [`inference::ActivationEstimator`] — run inference to collect activation stats

use crate::errors::{QuantizeError, Result};
#[cfg(feature = "calibration")]
use std::path::Path;

pub mod stats;
pub mod methods;
#[cfg(feature = "calibration")]
pub mod inference;

#[cfg(feature = "calibration")]
pub use inference::ActivationEstimator;

/// A collection of FP32 calibration samples used for range estimation.
#[derive(Clone)]
pub struct CalibrationDataset {
    /// Individual samples, each flattened to match `shape`.
    pub samples: Vec<Vec<f32>>,

    /// Shape of a single sample (excluding batch dimension).
    pub shape: Vec<usize>,
}

impl std::fmt::Debug for CalibrationDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalibrationDataset")
            .field("num_samples", &self.samples.len())
            .field("shape", &self.shape)
            .finish()
    }
}

impl CalibrationDataset {
    /// Load calibration samples from a NumPy `.npy` file.
    ///
    /// The array must be at least 2-dimensional `[batch, ...]`.
    ///
    /// Requires the `calibration` feature (enabled by default).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] if the file is missing, not `.npy`,
    /// or has an invalid shape.
    #[cfg(feature = "calibration")]
    pub fn from_numpy(path: impl AsRef<Path>) -> Result<Self> {
        use ndarray::{Array, IxDyn};

        let path = path.as_ref();

        if !path.exists() {
            return Err(QuantizeError::Calibration { reason: format!("File not found: {}", path.display()) });
        }

        let array: Array<f32, IxDyn> = if path.extension().and_then(|s| s.to_str()) == Some("npy") {
            ndarray_npy::read_npy(path)
                .map_err(|e| QuantizeError::Calibration { reason: format!("Failed to read NPY file '{}': {e}", path.display()) })?
        } else {
            return Err(QuantizeError::Calibration { reason: "Only .npy files supported currently".into() });
        };

        let shape: Vec<usize> = array.shape().to_vec();

        if shape.is_empty() {
            return Err(QuantizeError::Calibration { reason: "Invalid array shape".into() });
        }

        if shape.len() < 2 {
            return Err(QuantizeError::Calibration { reason: format!("Calibration data must be at least 2-dimensional (batch, ...). Got shape {:?}", shape) });
        }

        let num_samples = shape[0];
        let sample_size: usize = shape[1..].iter().product();

        let data = array.into_raw_vec();
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let start = i * sample_size;
            let end = start + sample_size;
            samples.push(data[start..end].to_vec());
        }

        Ok(Self {
            samples,
            shape: shape[1..].to_vec(),
        })
    }
    
    /// Generate random calibration samples uniformly distributed in `range`.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] if shape is empty, `num_samples` is 0,
    /// or the range is invalid.
    pub fn random(shape: Vec<usize>, num_samples: usize, range: (f32, f32)) -> Result<Self> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(QuantizeError::Calibration { reason: format!("Invalid shape: {:?} - all dimensions must be > 0", shape) });
        }
        if num_samples == 0 {
            return Err(QuantizeError::Calibration { reason: "num_samples must be > 0".into() });
        }
        if range.0 >= range.1 {
            return Err(QuantizeError::Calibration { reason: format!("Invalid range: ({}, {}) - min must be less than max", range.0, range.1) });
        }
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let sample_size: usize = shape.iter().product();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let sample: Vec<f32> = (0..sample_size)
                .map(|_| rng.gen_range(range.0..range.1))
                .collect();
            samples.push(sample);
        }

        Ok(Self {
            samples,
            shape,
        })
    }
    
    /// Create a dataset from pre-existing sample vectors.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] if `samples` is empty or any
    /// sample has the wrong length for the given `shape`.
    pub fn from_samples(samples: Vec<Vec<f32>>, shape: Vec<usize>) -> Result<Self> {
        let num_samples = samples.len();

        if num_samples == 0 {
            return Err(QuantizeError::Calibration { reason: "No samples provided".into() });
        }

        let expected_size: usize = shape.iter().product();

        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != expected_size {
                return Err(QuantizeError::Calibration {
                    reason: format!(
                        "Sample {} has size {} but expected {} (shape: {:?})",
                        i, sample.len(), expected_size, shape
                    ),
                });
            }
        }
        
        Ok(Self {
            samples,
            shape,
        })
    }
    
    /// Shape of a single sample (excluding batch dimension).
    pub fn sample_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_dataset() {
        let dataset = CalibrationDataset::random(vec![3, 224, 224], 10, (-1.0, 1.0)).unwrap();

        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.sample_shape(), &[3, 224, 224]);
        assert_eq!(dataset.samples[0].len(), 3 * 224 * 224);
    }
    
    #[test]
    fn test_from_samples() {
        let samples = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let dataset = CalibrationDataset::from_samples(samples, vec![3]).unwrap();
        assert_eq!(dataset.len(), 2);
    }
}