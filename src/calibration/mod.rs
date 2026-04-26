//! Calibration datasets and activation-based range estimation.
//!
//! - [`CalibrationDataset`] — load or generate calibration samples
//! - [`methods::CalibrationMethod`] — range optimization strategies
//! - [`stats::ActivationStats`] — incremental min/max/histogram tracker
//! - [`inference::ActivationEstimator`] — run inference to collect activation stats

use crate::errors::{QuantizeError, Result};
#[cfg(feature = "calibration")]
use std::path::Path;

#[cfg(feature = "calibration")]
pub mod inference;
pub mod methods;
pub mod stats;

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
            return Err(QuantizeError::Calibration {
                reason: format!("File not found: {}", path.display()),
            });
        }

        let array: Array<f32, IxDyn> = if path.extension().and_then(|s| s.to_str()) == Some("npy") {
            ndarray_npy::read_npy(path).map_err(|e| QuantizeError::Calibration {
                reason: format!("Failed to read NPY file '{}': {e}", path.display()),
            })?
        } else {
            return Err(QuantizeError::Calibration {
                reason: "Only .npy files supported currently".into(),
            });
        };

        let shape: Vec<usize> = array.shape().to_vec();

        if shape.is_empty() {
            return Err(QuantizeError::Calibration {
                reason: "Invalid array shape".into(),
            });
        }

        if shape.len() < 2 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Calibration data must be at least 2-dimensional (batch, ...). Got shape {:?}",
                    shape
                ),
            });
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

    /// Load calibration samples from a HuggingFace `.safetensors` file that
    /// contains exactly one tensor.
    ///
    /// The tensor must be f32 and at least 2-dimensional `[batch, ...]`.
    /// Requires the `safetensors-input` feature.
    ///
    /// For files with multiple named tensors, use
    /// [`from_safetensors_named`](Self::from_safetensors_named) instead.
    #[cfg(feature = "safetensors-input")]
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let buffer = std::fs::read(path).map_err(|e| QuantizeError::Calibration {
            reason: format!("Failed to read safetensors file '{}': {e}", path.display()),
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
            QuantizeError::Calibration {
                reason: format!("Failed to parse safetensors file: {e}"),
            }
        })?;
        let names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
        if names.is_empty() {
            return Err(QuantizeError::Calibration {
                reason: "safetensors file contains no tensors".into(),
            });
        }
        if names.len() > 1 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "safetensors file contains {} tensors; pass one explicitly via \
                     from_safetensors_named().  Available tensors: {}",
                    names.len(),
                    names.join(", ")
                ),
            });
        }
        Self::from_safetensors_view(&tensors, &names[0])
    }

    /// Load calibration samples from a specific named tensor inside a
    /// `.safetensors` file.
    ///
    /// Requires the `safetensors-input` feature.
    #[cfg(feature = "safetensors-input")]
    pub fn from_safetensors_named(path: impl AsRef<Path>, tensor_name: &str) -> Result<Self> {
        let path = path.as_ref();
        let buffer = std::fs::read(path).map_err(|e| QuantizeError::Calibration {
            reason: format!("Failed to read safetensors file '{}': {e}", path.display()),
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
            QuantizeError::Calibration {
                reason: format!("Failed to parse safetensors file: {e}"),
            }
        })?;
        Self::from_safetensors_view(&tensors, tensor_name)
    }

    #[cfg(feature = "safetensors-input")]
    fn from_safetensors_view(
        tensors: &safetensors::SafeTensors<'_>,
        tensor_name: &str,
    ) -> Result<Self> {
        use safetensors::Dtype;

        let view = tensors
            .tensor(tensor_name)
            .map_err(|e| QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' not found in safetensors file: {e}",
                    tensor_name
                ),
            })?;

        if view.dtype() != Dtype::F32 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' has dtype {:?}; only F32 is supported for calibration input",
                    tensor_name,
                    view.dtype()
                ),
            });
        }

        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() < 2 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Calibration tensor must be at least 2-dimensional (batch, ...). \
                     Got shape {:?}",
                    shape
                ),
            });
        }
        let expected_bytes: usize = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let raw = view.data();
        if raw.len() != expected_bytes {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' data size {} bytes does not match shape {:?} \
                     × 4 = {} bytes",
                    tensor_name,
                    raw.len(),
                    shape,
                    expected_bytes
                ),
            });
        }

        // safetensors stores data little-endian, which matches every target
        // quantize-rs builds on today.  Decode per-f32 explicitly to stay
        // endian-safe rather than relying on an unchecked cast.
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let num_samples = shape[0];
        let sample_size: usize = shape[1..].iter().product();
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
            return Err(QuantizeError::Calibration {
                reason: format!("Invalid shape: {:?} - all dimensions must be > 0", shape),
            });
        }
        if num_samples == 0 {
            return Err(QuantizeError::Calibration {
                reason: "num_samples must be > 0".into(),
            });
        }
        if range.0 >= range.1 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Invalid range: ({}, {}) - min must be less than max",
                    range.0, range.1
                ),
            });
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

        Ok(Self { samples, shape })
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
            return Err(QuantizeError::Calibration {
                reason: "No samples provided".into(),
            });
        }

        let expected_size: usize = shape.iter().product();

        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != expected_size {
                return Err(QuantizeError::Calibration {
                    reason: format!(
                        "Sample {} has size {} but expected {} (shape: {:?})",
                        i,
                        sample.len(),
                        expected_size,
                        shape
                    ),
                });
            }
        }

        Ok(Self { samples, shape })
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
        let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let dataset = CalibrationDataset::from_samples(samples, vec![3]).unwrap();
        assert_eq!(dataset.len(), 2);
    }

    #[cfg(feature = "safetensors-input")]
    #[test]
    fn test_from_safetensors_round_trip() {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        use std::collections::HashMap;

        // Build 3 samples of shape [2, 4] = 24 floats.
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let raw: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![3, 2, 4], &raw).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("input".to_string(), view);
        let bytes = serialize(&tensors, &None).unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".safetensors").unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let dataset = CalibrationDataset::from_safetensors(tmp.path()).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.sample_shape(), &[2, 4]);
        // Each sample holds 8 floats.
        assert_eq!(dataset.samples[0].len(), 8);
        // First float of sample 0 is 0.0, first of sample 1 is 0.8 (index 8 * 0.1).
        assert!((dataset.samples[0][0] - 0.0).abs() < 1e-6);
        assert!((dataset.samples[1][0] - 0.8).abs() < 1e-6);
    }

    #[cfg(feature = "safetensors-input")]
    #[test]
    fn test_from_safetensors_multi_tensor_errors_without_name() {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        use std::collections::HashMap;

        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let raw: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let v1 = TensorView::new(Dtype::F32, vec![2, 4], &raw).unwrap();
        let v2 = TensorView::new(Dtype::F32, vec![2, 4], &raw).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("a".to_string(), v1);
        tensors.insert("b".to_string(), v2);
        let bytes = serialize(&tensors, &None).unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".safetensors").unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let err = CalibrationDataset::from_safetensors(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("contains 2 tensors"));

        // But named access works.
        let dataset = CalibrationDataset::from_safetensors_named(tmp.path(), "a").unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.sample_shape(), &[4]);
    }
}
